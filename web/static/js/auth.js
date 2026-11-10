/* 登录 / 注册页 */
(function () {
  "use strict";

  function showInlineError(msg) {
    const el = document.getElementById("authError");
    if (el) {
      el.style.display = "block";
      el.textContent = msg;
    }
  }

  function boot() {
    if (!window.App || typeof App.api !== "function") {
      showInlineError("页面脚本加载失败，请按 Ctrl+F5 强制刷新后重试");
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const nextUrl = params.get("next") || "/";

    const loginForm = document.getElementById("loginForm");
    const registerForm = document.getElementById("registerForm");
    const roleTabs = document.getElementById("roleTabs");
    const roleInput = document.getElementById("role");
    const authCard = document.getElementById("authCard");

    function applyLoginTheme(role) {
      const theme = role === "admin" ? "theme-admin" : "theme-user";
      const root = document.documentElement;
      root.classList.remove("theme-user", "theme-admin");
      root.classList.add(theme);
      document.body.classList.remove("theme-user", "theme-admin");
      document.body.classList.add(theme);
      if (authCard) {
        authCard.classList.remove("theme-user", "theme-admin");
        authCard.classList.add(theme);
      }
    }

    if (roleTabs && roleInput) {
      applyLoginTheme(roleInput.value);
      roleTabs.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-role]");
        if (!btn) return;
        roleTabs.querySelectorAll(".role-tab").forEach((el) => el.classList.remove("active"));
        btn.classList.add("active");
        roleInput.value = btn.dataset.role;
        applyLoginTheme(btn.dataset.role);
      });
    }

    if (loginForm) {
      loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const submitBtn = document.getElementById("submitBtn");
        const prev = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = "登录中…";
        App.toast("正在登录…", "info");
        try {
          await App.api("/api/auth/login", {
            method: "POST",
            body: {
              username: document.getElementById("username").value.trim(),
              password: document.getElementById("password").value,
              role: roleInput ? roleInput.value : "user",
            },
          });
          App.toast("登录成功", "ok");
          window.location.href = nextUrl.startsWith("/") ? nextUrl : "/";
        } catch (err) {
          showInlineError(err.message || "登录失败");
          App.toast(err.message || "登录失败", "err");
        } finally {
          submitBtn.disabled = false;
          submitBtn.textContent = prev;
        }
      });
    }

    if (registerForm) {
      registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const pwd = document.getElementById("password").value;
        const pwd2 = document.getElementById("password2").value;
        if (pwd !== pwd2) {
          App.toast("两次密码不一致", "err");
          return;
        }
        const submitBtn = document.getElementById("submitBtn");
        submitBtn.disabled = true;
        try {
          await App.api("/api/auth/register", {
            method: "POST",
            body: {
              username: document.getElementById("username").value.trim(),
              display_name: document.getElementById("displayName").value.trim(),
              password: pwd,
            },
          });
          App.toast("注册成功，即将跳转登录", "ok");
          setTimeout(() => {
            window.location.href = "/login";
          }, 600);
        } catch (err) {
          App.toast(err.message, "err");
        } finally {
          submitBtn.disabled = false;
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
