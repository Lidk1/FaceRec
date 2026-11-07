/* 公共工具：接口调用、提示、摄像头封装、状态栏。 */
(function () {
  "use strict";

  const App = {
    threshold: 0.75,
    captureWidth: 640,

    /** 统一请求封装，自动解包 {code,msg,data} 并抛出友好错误 */
    async api(path, options = {}) {
      const opts = Object.assign({ headers: {} }, options);
      if (opts.body && !(opts.body instanceof FormData)) {
        opts.headers["Content-Type"] = "application/json";
        if (typeof opts.body !== "string") opts.body = JSON.stringify(opts.body);
      }
      let resp;
      try {
        resp = await fetch(path, opts);
      } catch (err) {
        throw new Error("网络连接失败，请检查服务是否启动");
      }
      let body;
      try {
        body = await resp.json();
      } catch (err) {
        throw new Error(`服务返回异常（HTTP ${resp.status}）`);
      }
      if (!resp.ok || (body.code !== undefined && body.code !== 200)) {
        const msg = body.msg || `请求失败（HTTP ${resp.status}）`;
        const error = new Error(msg);
        error.code = body.code;
        error.data = body.data;
        throw error;
      }
      return body.data;
    },

    toast(message, type = "info") {
      const wrap = document.getElementById("toastWrap");
      if (!wrap) return;
      const el = document.createElement("div");
      el.className = `toast ${type}`;
      el.textContent = message;
      wrap.appendChild(el);
      setTimeout(() => {
        el.style.opacity = "0";
        el.style.transition = "opacity .3s";
        setTimeout(() => el.remove(), 320);
      }, 3200);
    },

    fmtTime(text) {
      return text ? String(text).replace("T", " ").slice(0, 19) : "-";
    },

    /** 摄像头封装：启用、关闭、抓取 JPEG dataURL */
    class Camera {
      constructor(video, overlay) {
        this.video = video;
        this.overlay = overlay;
        this.stream = null;
        this.captureSize = { width: 0, height: 0 };
      }

      get active() {
        return !!(this.stream && this.stream.active);
      }

      async start() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("当前浏览器不支持摄像头（需 HTTPS 或 localhost 环境）");
        }
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false,
        });
        this.video.srcObject = this.stream;
        await this.video.play();
        return this.stream;
      }

      stop() {
        if (this.stream) {
          this.stream.getTracks().forEach((t) => t.stop());
          this.stream = null;
        }
        this.video.srcObject = null;
        this.clearOverlay();
      }

      /** 抓取当前帧，长边缩小到 captureWidth，返回 {dataUrl, width, height} */
      capture() {
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        if (!vw || !vh) throw new Error("摄像头尚未就绪");
        const scale = Math.min(1, App.captureWidth / vw);
        const width = Math.round(vw * scale);
        const height = Math.round(vh * scale);
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d").drawImage(this.video, 0, 0, width, height);
        this.captureSize = { width, height };
        return { dataUrl: canvas.toDataURL("image/jpeg", 0.88), width, height };
      }

      /** 把检测框画到覆盖层（按视频显示比例缩放） */
      drawBoxes(boxes) {
        const canvas = this.overlay;
        if (!canvas) return;
        const rect = this.video.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!boxes || !boxes.length) return;
        const sx = canvas.width / (this.captureSize.width || 1);
        const sy = canvas.height / (this.captureSize.height || 1);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#2ee6a8";
        ctx.font = "12px Microsoft YaHei";
        boxes.forEach((b) => {
          ctx.strokeRect(b.x * sx, b.y * sy, b.w * sx, b.h * sy);
        });
      }

      clearOverlay() {
        if (!this.overlay) return;
        this.overlay.getContext("2d").clearRect(0, 0, this.overlay.width, this.overlay.height);
      }
    },
  };

  /** 顶栏状态：模型是否就绪 + 特征库规模 */
  async function refreshStatus() {
    const dot = document.getElementById("statusDot");
    const text = document.getElementById("statusText");
    if (!dot || !text) return;
    try {
      const s = await App.api("/api/face/status");
      dot.className = "dot " + (s.ready ? "ok" : "err");
      text.textContent = s.ready
        ? `模型就绪 · ${s.users} 人 / ${s.features} 条特征`
        : "模型未就绪";
      App.threshold = s.threshold;
    } catch (err) {
      dot.className = "dot err";
      text.textContent = "服务不可用";
    }
  }

  App.refreshStatus = refreshStatus;
  document.addEventListener("DOMContentLoaded", refreshStatus);
  window.App = App;
})();
