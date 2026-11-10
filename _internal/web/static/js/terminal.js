/* 识别终端：摄像头预览、连续识别、上传图片兜底、结果与记录展示 */
(function () {
  "use strict";

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const placeholder = document.getElementById("placeholder");
  const cameraBtn = document.getElementById("cameraBtn");
  const scanBtn = document.getElementById("scanBtn");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadInput = document.getElementById("uploadInput");
  const liveBadge = document.getElementById("liveBadge");
  const autoLoop = document.getElementById("autoLoop");
  const camera = new App.Camera(video, overlay);

  let scanning = false;
  let timer = null;
  let busy = false;
  let lastResult = null;

  function setHero(kind, state, name, detail) {
    const hero = document.getElementById("resultHero");
    hero.className = "result-hero " + kind;
    document.getElementById("resultState").textContent = state;
    document.getElementById("resultName").textContent = name;
    document.getElementById("resultUid").textContent = detail || "";
  }

  function renderResult(data) {
    lastResult = data;
    document.getElementById("resultHint").textContent = App.fmtTime(new Date().toISOString());
    document.getElementById("mSimilarity").textContent =
      data.similarity === null || data.similarity === undefined ? "—" : Number(data.similarity).toFixed(3);
    document.getElementById("mThreshold").textContent = Number(data.threshold).toFixed(2);
    document.getElementById("mCost").textContent = data.cost_ms + " ms";
    document.getElementById("mFaces").textContent = data.face_count;

    if (data.passed) {
      setHero("success", "识别成功", data.user_name, `${data.user_no} · 相似度 ${Number(data.similarity).toFixed(3)}`);
      camera.drawBoxes(data.boxes);
      App.toast(`欢迎，${data.user_name}`, "ok");
    } else {
      setHero("fail", data.result_text || "识别失败", data.message || "请重试",
        `相似度 ${data.similarity === null ? "—" : Number(data.similarity).toFixed(3)} · ${data.result_text}`);
      camera.clearOverlay();
    }
    loadRecords();
  }

  async function recognizeOnce(dataUrl) {
    const data = await App.api("/api/face/recognize", {
      method: "POST",
      body: { image: dataUrl },
    });
    renderResult(data);
    return data;
  }

  async function tick() {
    if (!scanning || busy || !camera.active) return;
    busy = true;
    try {
      const shot = camera.capture();
      await recognizeOnce(shot.dataUrl);
    } catch (err) {
      if (err.code !== 1005) {
        // 超时由服务端计为识别失败，这里只提示其它异常
        App.toast(err.message, "err");
      }
    } finally {
      busy = false;
    }
  }

  function startLoop() {
    stopLoop();
    if (!autoLoop.checked) return;
    timer = setInterval(tick, 1200);
  }

  function stopLoop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function setScanning(value) {
    scanning = value;
    scanBtn.textContent = value ? "停止识别" : "开始识别";
    scanBtn.classList.toggle("primary", !value);
    liveBadge.textContent = value ? "识别中" : camera.active ? "摄像头已就绪" : "摄像头未启用";
    liveBadge.classList.toggle("on", value);
    if (value) {
      startLoop();
      tick();
    } else {
      stopLoop();
      camera.clearOverlay();
    }
  }

  cameraBtn.addEventListener("click", async () => {
    if (camera.active) {
      setScanning(false);
      camera.stop();
      cameraBtn.textContent = "启用摄像头";
      scanBtn.disabled = true;
      uploadBtn.disabled = false;
      placeholder.style.display = "grid";
      placeholder.innerHTML = "摄像头已关闭<br>可重新启用，或使用「上传图片识别」";
      liveBadge.textContent = "摄像头未启用";
      liveBadge.classList.remove("on");
      return;
    }
    try {
      cameraBtn.disabled = true;
      await camera.start();
      placeholder.style.display = "none";
      cameraBtn.textContent = "关闭摄像头";
      scanBtn.disabled = false;
      uploadBtn.disabled = false;
      liveBadge.textContent = "摄像头已就绪";
      liveBadge.classList.add("on");
      document.getElementById("engineHint").textContent = "已获取摄像头视频流";
    } catch (err) {
      App.toast(err.message || "摄像头启用失败", "err");
      placeholder.innerHTML = "无法访问摄像头<br>请检查浏览器权限或设备连接";
    } finally {
      cameraBtn.disabled = false;
    }
  });

  scanBtn.addEventListener("click", () => {
    if (!camera.active) {
      App.toast("请先启用摄像头", "err");
      return;
    }
    setScanning(!scanning);
  });

  uploadInput.addEventListener("change", async () => {
    const file = uploadInput.files && uploadInput.files[0];
    if (!file) return;
    uploadBtn.disabled = true;
    try {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("图片读取失败"));
        reader.readAsDataURL(file);
      });
      // 上传图片走一次性识别，不启动摄像头
      const data = await App.api("/api/face/recognize", { method: "POST", body: { image: dataUrl } });
      renderResult(data);
      if (data.passed) {
        placeholder.style.display = "grid";
        placeholder.innerHTML = `已识别：${data.user_name}<br>可继续上传或启用摄像头`;
      }
    } catch (err) {
      App.toast(err.message, "err");
    } finally {
      uploadBtn.disabled = false;
      uploadInput.value = "";
    }
  });

  async function loadRecords() {
    try {
      const data = await App.api("/api/face/records?limit=8");
      const list = document.getElementById("recordList");
      if (!data.items.length) {
        list.innerHTML = '<div class="empty">暂无记录</div>';
        return;
      }
      list.innerHTML = data.items
        .map((r) => {
          const ok = r.result_type === 1;
          return `<div class="list-item">
            <div>
              <div>${r.user_name ? r.user_name + " · " + r.user_no : "未匹配"}</div>
              <div class="meta">${App.fmtTime(r.recognize_time)} · ${r.cost_ms}ms · 相似度 ${
            r.similarity === null ? "—" : Number(r.similarity).toFixed(3)
          }</div>
            </div>
            <span class="tag ${ok ? "ok" : "bad"}">${r.result_text}</span>
          </div>`;
        })
        .join("");
    } catch (err) {
      /* 状态栏已提示服务异常，这里静默 */
    }
  }

  document.getElementById("refreshRecords").addEventListener("click", loadRecords);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopLoop();
    } else if (scanning && autoLoop.checked) {
      startLoop();
    }
  });

  loadRecords();
  uploadBtn.disabled = false;
  App.refreshStatus();
})();
