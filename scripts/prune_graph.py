"""把 FaceNet 训练图裁剪为纯推理图（无需安装 TensorFlow）。

背景
----
公开的 ``20180402-114759.pb`` 是训练/导出阶段的计算图：

* 图里保留了训练输入队列 ``batch_join/fifo_queue``（FIFOQueueV2）；
* 批归一化用 ``tf.cond(phase_train, ...)`` 生成训练/推理两个分支，
  图上因此存在 2019 个 ``Switch`` 与 449 个 ``Merge``，训练分支里还有
  ``random_uniform`` 等推理后端无法解析的算子。

OpenCV DNN 在导入时会实例化图中的全部节点，所以原始 .pb 无法直接加载。
本脚本在字节级解析 GraphDef，按 ``phase_train = False`` 折叠 cond 分支，
再裁剪掉输出 ``embeddings`` 依赖不到的节点，得到纯推理图。

产物
----
``facenet2rec-inference.pb``：输入 ``input``（1×160×160×3，RGB，
按 (x-127.5)/128 标准化），输出 ``embeddings``（1×512，L2 归一化）。

用法::

    python scripts/prune_graph.py model/20180402-114759.pb \
        model/facenet2rec-inference.pb
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# --------------------------------------------------------------------------
# 极简 protobuf 编解码（只覆盖 GraphDef / NodeDef 需要的 wire type）
# --------------------------------------------------------------------------

WIRE_VARINT, WIRE_64BIT, WIRE_LEN, WIRE_32BIT = 0, 1, 2, 5


def read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def write_varint(value: int) -> bytes:
    value &= (1 << 64) - 1  # int64 负数按补码编码
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_field(field: int, wire: int, value) -> bytes:
    key = write_varint((field << 3) | wire)
    if wire == WIRE_VARINT:
        return key + write_varint(int(value))
    if wire == WIRE_LEN:
        return key + write_varint(len(value)) + value
    raise ValueError(f"unsupported wire type {wire}")


def iter_fields(buf: bytes):
    """逐字段产出 (field, wire, value)。

    varint 字段的 value 为 int，length-delimited 字段为 bytes 切片。
    """
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == WIRE_VARINT:
            value, pos = read_varint(buf, pos)
        elif wire == WIRE_LEN:
            length, pos = read_varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == WIRE_64BIT:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == WIRE_32BIT:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


# --------------------------------------------------------------------------
# NodeDef 读写
# --------------------------------------------------------------------------


class Node:
    """一个 NodeDef：name / op / inputs 可写，其余字段按原字节保留。"""

    __slots__ = ("fields", "name", "op", "inputs")

    def __init__(self, blob: bytes) -> None:
        self.fields: List[Tuple[int, int, object]] = list(iter_fields(blob))
        self.name = ""
        self.op = ""
        self.inputs: List[str] = []
        for field, wire, value in self.fields:
            if field == 1 and wire == WIRE_LEN:
                self.name = value.decode("utf-8", "replace")
            elif field == 2 and wire == WIRE_LEN:
                self.op = value.decode("utf-8", "replace")
            elif field == 3 and wire == WIRE_LEN:
                self.inputs.append(value.decode("utf-8", "replace"))

    def to_bytes(self) -> bytes:
        return b"".join(encode_field(field, wire, value) for field, wire, value in self.fields)


def set_node_inputs(node: Node, inputs: List[str]) -> None:
    """用新的 input 列表替换 NodeDef 中的 input 字段。"""
    payloads = [item.encode("utf-8") for item in inputs]
    rebuilt: List[Tuple[int, int, object]] = []
    inserted = False
    for field, wire, value in node.fields:
        if field == 3:
            if not inserted:
                rebuilt.extend((3, WIRE_LEN, payload) for payload in payloads)
                inserted = True
            continue
        rebuilt.append((field, wire, value))
    if not inserted:
        rebuilt.extend((3, WIRE_LEN, payload) for payload in payloads)
    node.fields = rebuilt
    node.inputs = list(inputs)


def parse_graph(data: bytes) -> Tuple[List[Node], List[Tuple[int, int, object]]]:
    nodes: List[Node] = []
    others: List[Tuple[int, int, object]] = []
    for field, wire, value in iter_fields(data):
        if field == 1 and wire == WIRE_LEN:
            nodes.append(Node(value))
        else:
            others.append((field, wire, value))
    return nodes, others


def write_graph(nodes: Iterable[Node], others: List[Tuple[int, int, object]]) -> bytes:
    out = bytearray()
    for node in nodes:
        out += encode_field(1, WIRE_LEN, node.to_bytes())
    for field, wire, value in others:
        out += encode_field(field, wire, value)
    return bytes(out)


# --------------------------------------------------------------------------
# 图裁剪
# --------------------------------------------------------------------------

PHASE_PREDICATES = {"phase_train"}


def tensor_name(ref: str) -> Tuple[str, str]:
    """把 "node:1" / "^node" 拆成 (节点名, 端口或控制标记)。"""
    control = ref.startswith("^")
    ref = ref.lstrip("^")
    if ":" in ref:
        base, port = ref.rsplit(":", 1)
        return base, ("^" if control else port)
    return ref, ("^" if control else "")


def join_ref(base: str, port: str) -> str:
    if port == "^":
        return "^" + base
    return f"{base}:{port}" if port else base


def _identity_map(nodes: List[Node]) -> Dict[str, str]:
    """收集 predicate 的 Identity 别名（phase_train -> *_/pred_id）。"""
    aliases: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.op == "Identity" and len(node.inputs) == 1:
                src, _ = tensor_name(node.inputs[0])
                if src in PHASE_PREDICATES | set(aliases) and node.name not in aliases:
                    aliases[node.name] = src
                    changed = True
    return aliases


def make_placeholder_node(name: str, dims: List[int]) -> Node:
    """构造一个 Placeholder NodeDef（float32，指定 NHWC 形状）。"""
    # AttrValue { type: DT_FLOAT(1) }
    dtype_attr = encode_field(5, WIRE_VARINT, 1)
    # TensorShapeProto { dim { size } ... }
    shape_proto = b"".join(
        encode_field(2, WIRE_LEN, encode_field(1, WIRE_VARINT, int(dim))) for dim in dims
    )
    shape_attr = encode_field(6, WIRE_LEN, shape_proto)
    blob = bytearray()
    blob += encode_field(1, WIRE_LEN, name.encode("utf-8"))
    blob += encode_field(2, WIRE_LEN, b"Placeholder")
    for key, attr in (("dtype", dtype_attr), ("shape", shape_attr)):
        entry = encode_field(1, WIRE_LEN, key.encode("utf-8")) + encode_field(2, WIRE_LEN, attr)
        blob += encode_field(5, WIRE_LEN, entry)
    return Node(bytes(blob))


def prune(graph_bytes: bytes, keep_outputs: Iterable[str] = ("embeddings",)) -> Tuple[bytes, dict]:
    nodes, others = parse_graph(graph_bytes)
    by_name = {node.name: node for node in nodes}
    aliases = _identity_map(nodes)

    def is_false_predicate(ref: str) -> bool:
        base, _ = tensor_name(ref)
        return base in PHASE_PREDICATES or base in aliases

    # 1) 收集 Switch：pred 恒为 False 时，端口 0 是活分支，端口 1 是死分支
    switch_data: Dict[str, str] = {}
    for node in nodes:
        if node.op != "Switch" or len(node.inputs) != 2:
            continue
        data_ref, pred_ref = node.inputs
        if is_false_predicate(pred_ref):
            switch_data[node.name] = data_ref

    # 2) 死分支传播：任一数据输入来自死分支则该节点不可达；
    #    Merge 只有在全部输入都不可达时才不可达。
    def ref_is_dead(ref: str, dead: set) -> bool:
        base, port = tensor_name(ref)
        if ref.startswith("^"):
            return base in dead
        if base in switch_data and port == "1":
            return True
        return base in dead

    dead: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.name in dead:
                continue
            if node.op == "Switch" and node.name in switch_data:
                continue
            if node.op == "Merge":
                if all(ref_is_dead(ref, dead) for ref in node.inputs):
                    dead.add(node.name)
                    changed = True
                continue
            if any(ref_is_dead(ref, dead) for ref in node.inputs):
                dead.add(node.name)
                changed = True

    # 3) 重写活节点输入：Switch:0 直连数据，Merge 只保留活输入
    for node in nodes:
        if node.name in dead:
            continue
        if node.op == "Switch" and node.name in switch_data:
            continue
        rewritten: List[str] = []
        for ref in node.inputs:
            base, port = tensor_name(ref)
            if base in switch_data and port != "1":
                src = switch_data[base]
                src_base, src_port = tensor_name(src)
                rewritten.append(join_ref(src_base, port if port not in ("", "0") else src_port))
            else:
                rewritten.append(ref)
        if node.op == "Merge":
            rewritten = [ref for ref in rewritten if not ref_is_dead(ref, dead)]
        if rewritten != node.inputs:
            set_node_inputs(node, rewritten)

    # 4) Merge 折叠：只剩一个活输入时，把消费者直接接到该输入
    merge_source: Dict[str, str] = {}
    for node in nodes:
        if node.op == "Merge" and node.name not in dead and len(node.inputs) == 1:
            merge_source[node.name] = node.inputs[0]

    def resolve(ref: str) -> str:
        """沿 Merge 折叠链一路解析到真正的数据来源。"""
        base, port = tensor_name(ref)
        seen = set()
        while base in merge_source and base not in seen:
            seen.add(base)
            src = merge_source[base]
            src_base, src_port = tensor_name(src)
            base = src_base
            if port in ("", "0"):
                port = src_port
        return join_ref(base, port)

    if merge_source:
        for node in nodes:
            if node.name in dead or node.name in merge_source:
                continue
            rewritten = [resolve(ref) for ref in node.inputs]
            if rewritten != node.inputs:
                set_node_inputs(node, rewritten)
    dead.update(merge_source.keys())

    # 5) 重建模型输入：原图通过 batch_join 队列产生 image_batch，
    #    推理图没有输入可用，这里把 "input" 换成真正的 Placeholder。
    input_node = by_name.get("input")
    if input_node is not None and input_node.op != "Placeholder":
        new_input = make_placeholder_node("input", [-1, 160, 160, 3])
        nodes = [new_input if node.name == "input" else node for node in nodes]
        by_name["input"] = new_input

    # 6) 依赖闭包：只保留 outputs 依赖到的节点（忽略控制依赖，
    #    控制边通常指向训练期算子，推理阶段无意义）
    keep = set(keep_outputs) | {"input"}
    frontier = list(keep)
    while frontier:
        name = frontier.pop()
        node = by_name.get(name)
        if node is None:
            continue
        for ref in node.inputs:
            base, port = tensor_name(ref)
            if port == "^":
                continue
            if base not in keep:
                keep.add(base)
                frontier.append(base)
    keep -= dead

    kept_nodes = [node for node in nodes if node.name in keep]
    for node in kept_nodes:
        # 清理指向已删除节点的控制依赖
        filtered = [
            ref for ref in node.inputs
            if not (ref.startswith("^") and tensor_name(ref)[0] not in keep)
        ]
        if filtered != node.inputs:
            set_node_inputs(node, filtered)

    out = write_graph(kept_nodes, others)
    op_counts: Dict[str, int] = {}
    for node in kept_nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    leftover = {op: cnt for op, cnt in op_counts.items()
                if "Queue" in op or op in {"Switch", "Merge", "Assign", "RandomUniform"}}
    summary = {
        "nodes_in": len(nodes),
        "nodes_out": len(kept_nodes),
        "switches_folded": len(switch_data),
        "merges_folded": len(merge_source),
        "dead_marked": len(dead),
        "leftover_training_ops": leftover,
        "ops": op_counts,
    }
    return out, summary


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    data = src.read_bytes()
    pruned, summary = prune(data)
    dst.write_bytes(pruned)
    print(f"source : {src}  ({len(data)/1048576:.1f} MB, {summary['nodes_in']} nodes)")
    print(f"output : {dst}  ({len(pruned)/1048576:.1f} MB, {summary['nodes_out']} nodes)")
    print(
        "折叠: Switch={switches_folded} Merge={merges_folded} 死节点={dead_marked}".format(**summary)
    )
    if summary["leftover_training_ops"]:
        print(f"警告: 仍残留训练算子 {summary['leftover_training_ops']}")
    top = sorted(summary["ops"].items(), key=lambda kv: -kv[1])[:8]
    print("算子分布:", ", ".join(f"{op}={cnt}" for op, cnt in top))


if __name__ == "__main__":
    main()
