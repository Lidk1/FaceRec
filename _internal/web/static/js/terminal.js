/* 识别终端：摄像头预览、连续识别、上传图片兜底、结果与记录展示 */
(function () {
  "use strict";

  function boot() {
    if (!window.App || !App.Camera) {
      const ph = document.getElementById("placeholder");
      if (ph) {
        ph.innerHTML =
          "页面脚本未加载完整<br>请确认已用 http://127.0.0.1:8080 打开，并试 Ctrl+F5 强制刷新";
      }
      return;
    }

    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const placeholder = document.getElementById("placeholder");
    const cameraBtn = document.getElementById("cameraBtn");
    const scanBtn = document.getElementById("scanBtn");
    const uploadBtn = document.getElementById("uploadBtn");
    const uploadInput = document.getElementById("uploadInput");
    const liveBadge = document.getElementById("liveBadge");
    const autoLoop = document.getElementById("autoLoop");

    if (!video || !cameraBtn) {
      App.toast("页面结构异常，请刷新浏览器", "err");
      return;
    }

    const camera = new App.Camera(video, overlay);

    let scanning = false;
    let timer = null;
    let busy = false;

    function setHero(kind, state, name, detail) {
      const hero = document.getElementById("resultHero");
      hero.className = "result-hero " + kind;
      document.getElementById("resultState").textContent = state;
      document.getElementById("resultName").textContent = name;
      document.getElementById("resultUid").textContent = detail || "";
    }

    function renderResult(data) {
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
        if (err.code !== 1005) App.toast(err.message, "err");
      } finally {
        busy = false;
      }
    }

    function startLoop() {
      if (timer) clearInterval(timer);
      timer = null;
      if (!autoLoop || !autoLoop.checked) return;
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
        placeholder.style.display = "grid";
        placeholder.innerHTML = "摄像头已关闭<br>可重新启用，或使用「上传图片识别」";
        liveBadge.textContent = "摄像头未启用";
        liveBadge.classList.remove("on");
        return;
      }

      const prevLabel = cameraBtn.textContent;
      try {
        cameraBtn.disabled = true;
        cameraBtn.textContent = "正在打开…";
        App.toast("正在请求摄像头权限，请在浏览器弹窗中选择「允许」", "info");
        if (placeholder) placeholder.innerHTML = "等待摄像头授权…";

        await camera.start();

        if (placeholder) placeholder.style.display = "none";
        cameraBtn.textContent = "关闭摄像头";
        scanBtn.disabled = false;
        if (uploadBtn) uploadBtn.disabled = false;
        liveBadge.textContent = "摄像头已就绪";
        liveBadge.classList.add("on");
        const hint = document.getElementById("engineHint");
        if (hint) hint.textContent = "已获取摄像头视频流";
        App.toast("摄像头已开启", "ok");
      } catch (err) {
        cameraBtn.textContent = prevLabel;
        App.toast(err.message || "摄像头启用失败", "err");
        if (placeholder) {
          placeholder.style.display = "grid";
          placeholder.innerHTML = "无法访问摄像头<br>" + (err.message || "请检查权限与设备");
        }
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

    if (uploadBtn && uploadInput) {
      uploadBtn.addEventListener("click", () => {
        if (uploadBtn.disabled) return;
        uploadInput.click();
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
          const data = await App.api("/api/face/recognize", { method: "POST", body: { image: dataUrl } });
          renderResult(data);
          if (data.passed && placeholder) {
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
    }

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
        /* 静默 */
      }
    }

    const refreshRecords = document.getElementById("refreshRecords");
    if (refreshRecords) refreshRecords.addEventListener("click", loadRecords);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopLoop();
      else if (scanning && autoLoop && autoLoop.checked) startLoop();
    });

    loadRecords();
    if (uploadBtn) uploadBtn.disabled = false;
    App.refreshStatus();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
