@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [FaceRec] 使用源码启动（含登录/注册）。若缺少依赖请执行: pip install -r requirements.txt
if not exist "model\facenet2rec-inference.pb" (
  if exist "_internal\model\facenet2rec-inference.pb" (
    if not exist "model" mkdir "model"
    copy /Y "_internal\model\facenet2rec-inference.pb" "model\" >nul
    echo 已从 _internal 复制模型到 model\
  )
)
python run.py --no-browser %*
pause
