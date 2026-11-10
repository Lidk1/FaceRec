/* 人脸注册页：摄像头采集 / 本地照片，多帧提交，展示注册结果 */
(function () {
  "use strict";

  const MAX = 5;
  const video = document.getElementById("video");
  const placeholder = document.getElementById("placeholder");
  const cameraBtn = document.getElementById("cameraBtn");
  const captureBtn = document.getElementById("captureBtn");
  const pickBtn = document.getElementById("pickBtn");
  const fileInput = document.getElementById("fileInput");
  const enrollBtn = document.getElementById("enrollBtn");
  const clearBtn = document.getElementById("clearBtn");
  const thumbs = document.getElementById("thumbs");
  const camera = new App.Camera(video, null);

  let samples = []; // [{dataUrl, source}]

  function refreshUi() {
    document.getElementById("captureHint").textContent = `已采集 ${samples.length} / ${MAX} 帧`;
    enrollBtn.disabled = samples.length === 0;
    clearBtn.disabled = samples.length === 0;
    if (!samples.length) {
      thumbs.innerHTML = '<div class="empty">尚未采集样本</div>';
      return;
    }
    thumbs.innerHTML = samples
      .map(
        (s, i) => `<div class="thumb" title="${s.source}">
          <img src="${s.dataUrl}" alt="样本${i + 1}">
          <button data-index="${i}" title="移除">×</button>
        </div>`
      )
      .join("");
  }

  function addSample(dataUrl, source) {
    if (samples.length >= MAX) {
      App.toast(`最多采集 ${MAX} 帧样本`, "err");
      return;
    }
    samples.push({ dataUrl, source });
    refreshUi();
  }

  thumbs.addEventListener("click", (event) => {
    const index = event.target.getAttribute("data-index");
    if (index === null) return;
    samples.splice(Number(index), 1);
    refreshUi();
  });

  cameraBtn.addEventListener("click", async () => {
    if (camera.active) {
      camera.stop();
      cameraBtn.textContent = "启用摄像头";
      captureBtn.disabled = true;
      placeholder.style.display = "grid";
      placeholder.innerHTML = "摄像头已关闭<br>可重新启用，或选择本地照片";
      document.getElementById("liveBadge").textContent = "摄像头未启用";
      return;
    }
    const prevLabel = cameraBtn.textContent;
    try {
      cameraBtn.disabled = true;
      cameraBtn.textContent = "正在打开…";
      App.toast("正在请求摄像头权限，请在浏览器弹窗中选择「允许」", "info");
      placeholder.innerHTML = "等待摄像头授权…";
      await camera.start();
      placeholder.style.display = "none";
      cameraBtn.textContent = "关闭摄像头";
      captureBtn.disabled = false;
      document.getElementById("liveBadge").textContent = "摄像头已就绪";
      document.getElementById("liveBadge").classList.add("on");
      App.toast("摄像头已开启", "ok");
    } catch (err) {
      cameraBtn.textContent = prevLabel;
      App.toast(err.message, "err");
      placeholder.style.display = "grid";
      placeholder.innerHTML = "无法访问摄像头<br>" + (err.message || "请检查浏览器权限，或改用本地照片");
    } finally {
      cameraBtn.disabled = false;
    }
  });

  captureBtn.addEventListener("click", () => {
    try {
      const shot = camera.capture();
      addSample(shot.dataUrl, "摄像头采集");
    } catch (err) {
      App.toast(err.message, "err");
    }
  });

  pickBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", async () => {
    const files = Array.from(fileInput.files || []);
    for (const file of files) {
      if (samples.length >= MAX) break;
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("图片读取失败"));
        reader.readAsDataURL(file);
      });
      addSample(dataUrl, file.name);
    }
    fileInput.value = "";
  });

  clearBtn.addEventListener("click", () => {
    samples = [];
    refreshUi();
    setHero("idle", "尚未提交", "—", "采集完成后点击「提交注册」");
  });

  function setHero(kind, state, name, detail) {
    const hero = document.getElementById("resultHero");
    hero.className = "result-hero " + kind;
    document.getElementById("resultState").textContent = state;
    document.getElementById("resultName").textContent = name;
    document.getElementById("resultUid").textContent = detail || "";
  }

  enrollBtn.addEventListener("click", async () => {
    const userNo = document.getElementById("userNo").value.trim();
    const userName = document.getElementById("userName").value.trim();
    if (!userNo || !userName) {
      App.toast("请填写学号/工号与姓名", "err");
      return;
    }
    if (!samples.length) {
      App.toast("请至少采集 1 帧人脸", "err");
      return;
    }
    enrollBtn.disabled = true;
    enrollBtn.textContent = "提交中…";
    try {
      const data = await App.api("/api/face/enroll", {
        method: "POST",
        body: {
          user_no: userNo,
          user_name: userName,
          dept: document.getElementById("dept").value.trim() || null,
          phone: document.getElementById("phone").value.trim() || null,
          replace: document.getElementById("replace").checked,
          images: samples.map((s) => s.dataUrl),
        },
      });
      setHero("success", "注册成功", data.user_name, `${data.user_no} · 用户 ID ${data.user_id}`);
      document.getElementById("mSamples").textContent = data.samples;
      document.getElementById("mQuality").textContent = Number(data.avg_quality).toFixed(3);
      document.getElementById("mCost").textContent = data.cost_ms + " ms";
      document.getElementById("mUserId").textContent = data.user_id;
      const rejected = data.rejected || [];
      if (rejected.length) {
        document.getElementById("sampleTip").textContent =
          `已自动拒绝 ${rejected.length} 帧：` + rejected.map((r) => r.message).join("；");
      }
      App.toast(`注册成功，入库 ${data.samples} 帧特征`, "ok");
      App.refreshStatus();
      samples = [];
      refreshUi();
      document.getElementById("enrollForm").reset();
    } catch (err) {
      setHero("fail", "注册失败", err.message, `错误码 ${err.code || "-"}`);
      App.toast(err.message, "err");
    } finally {
      enrollBtn.disabled = samples.length === 0;
      enrollBtn.textContent = "提交注册";
    }
  });

  refreshUi();
  App.refreshStatus();
})();
