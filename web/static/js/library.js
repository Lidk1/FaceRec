/* 特征库管理：用户列表、特征明细、删除特征 / 注销用户、识别统计 */
(function () {
  "use strict";

  const rows = document.getElementById("userRows");
  const featureDetail = document.getElementById("featureDetail");

  function esc(text) {
    return String(text === null || text === undefined ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function loadStats() {
    try {
      const s = await App.api("/api/face/stats");
      document.getElementById("sUsers").textContent = s.users;
      document.getElementById("sFeatures").textContent = s.features;
      document.getElementById("sRate").textContent = (Number(s.success_rate) * 100).toFixed(1) + "%";
      document.getElementById("recordCount").textContent = `共 ${s.total} 次`;

      const items = Object.entries(s.by_result_text || {}).filter(([k]) => k !== "识别成功");
      document.getElementById("failList").innerHTML = items.length
        ? items
            .map(
              ([name, count]) =>
                `<div class="list-item"><span>${esc(name)}</span><span class="tag bad">${count} 次</span></div>`
            )
            .join("")
        : '<div class="empty">暂无失败记录</div>';
    } catch (err) {
      App.toast(err.message, "err");
    }
  }

  async function loadUsers(keyword = "") {
    rows.innerHTML = '<tr><td colspan="6" class="empty">加载中…</td></tr>';
    try {
      const data = await App.api("/api/face/users?limit=100&keyword=" + encodeURIComponent(keyword));
      document.getElementById("userCount").textContent = `共 ${data.total} 人`;
      if (!data.items.length) {
        rows.innerHTML = '<tr><td colspan="6" class="empty">暂无用户，请先完成人脸注册</td></tr>';
        return;
      }
      rows.innerHTML = data.items
        .map(
          (u) => `<tr>
            <td>${esc(u.user_no)}</td>
            <td>${esc(u.user_name)}</td>
            <td>${esc(u.dept || "—")}</td>
            <td><span class="pill ${u.feature_count ? "ok" : "warn"}">${u.feature_count} 条</span></td>
            <td class="muted">${App.fmtTime(u.create_time)}</td>
            <td>
              <button class="btn sm ghost" data-act="detail" data-id="${u.user_id}">查看</button>
              <button class="btn sm danger" data-act="del-feature" data-id="${u.user_id}">删特征</button>
              <button class="btn sm danger" data-act="del-user" data-id="${u.user_id}" data-no="${esc(u.user_no)}">注销</button>
            </td>
          </tr>`
        )
        .join("");
    } catch (err) {
      rows.innerHTML = `<tr><td colspan="6" class="empty">${esc(err.message)}</td></tr>`;
    }
  }

  async function showDetail(userId) {
    try {
      const u = await App.api(`/api/face/users/${userId}`);
      const features = u.features || [];
      featureDetail.innerHTML = `
        <div class="list-item" style="align-items:flex-start">
          <div>
            <div><strong>${esc(u.user_name)}</strong> · ${esc(u.user_no)} · ${esc(u.dept || "—")}</div>
            <div class="meta">用户 ID ${u.user_id} · 注册于 ${App.fmtTime(u.create_time)}</div>
            <div class="mt small muted">特征明细（仅展示元数据，密文不对外输出）</div>
            ${
              features.length
                ? features
                    .map(
                      (f) =>
                        `<div class="meta">样本 #${f.feature_no} · ${f.feature_dim} 维 · 质量 ${Number(
                          f.quality_score || 0
                        ).toFixed(3)} · ${esc(f.model_version)}</div>`
                    )
                    .join("")
                : '<div class="meta">暂无有效特征</div>'
            }
          </div>
          <button class="btn sm ghost" id="closeDetail">收起</button>
        </div>`;
      document.getElementById("closeDetail").onclick = () => (featureDetail.innerHTML = "");
    } catch (err) {
      App.toast(err.message, "err");
    }
  }

  async function loadRecords() {
    try {
      const data = await App.api("/api/face/records?limit=12");
      const list = document.getElementById("recordList");
      list.innerHTML = data.items.length
        ? data.items
            .map((r) => {
              const ok = r.result_type === 1;
              return `<div class="list-item">
                <div>
                  <div>${r.user_name ? esc(r.user_name) + " · " + esc(r.user_no) : "未匹配"}</div>
                  <div class="meta">${App.fmtTime(r.recognize_time)} · ${r.cost_ms}ms</div>
                </div>
                <span class="tag ${ok ? "ok" : "bad"}">${esc(r.result_text)}</span>
              </div>`;
            })
            .join("")
        : '<div class="empty">暂无记录</div>';
    } catch (err) {
      /* 忽略：统计卡片已提示 */
    }
  }

  rows.addEventListener("click", async (event) => {
    const btn = event.target.closest("button[data-act]");
    if (!btn) return;
    const act = btn.dataset.act;
    const userId = btn.dataset.id;
    if (act === "detail") {
      showDetail(userId);
      return;
    }
    if (act === "del-feature") {
      if (!confirm("确认删除该用户的全部人脸特征？删除后需重新注册才能刷脸。")) return;
      try {
        const data = await App.api(`/api/face/users/${userId}/features`, { method: "DELETE" });
        App.toast(`已删除 ${data.deleted} 条特征`, "ok");
        loadUsers(document.getElementById("search").value.trim());
        loadStats();
        featureDetail.innerHTML = "";
      } catch (err) {
        App.toast(err.message, "err");
      }
      return;
    }
    if (act === "del-user") {
      if (!confirm(`确认注销用户 ${btn.dataset.no}？其账户与特征将一并删除，识别记录保留。`)) return;
      try {
        await App.api(`/api/face/users/${userId}`, { method: "DELETE" });
        App.toast("用户已注销", "ok");
        loadUsers(document.getElementById("search").value.trim());
        loadStats();
        featureDetail.innerHTML = "";
      } catch (err) {
        App.toast(err.message, "err");
      }
    }
  });

  document.getElementById("refreshBtn").addEventListener("click", () => {
    loadUsers(document.getElementById("search").value.trim());
    loadStats();
    loadRecords();
    App.refreshStatus();
  });

  document.getElementById("search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadUsers(event.target.value.trim());
  });

  loadUsers();
  loadStats();
  loadRecords();
  App.refreshStatus();
})();
