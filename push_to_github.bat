@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === 正在检查本地 Git 提交历史 ===
git log --oneline
echo.
echo === 正在推送到 https://github.com/plmnbvcxz46/steamsales-guess.git ===
git remote set-url origin https://github.com/plmnbvcxz46/steamsales-guess.git
git branch -M main
git push -u origin main --force
echo === 推送完成！请访问 https://github.com/plmnbvcxz46/steamsales-guess 查看 ===
pause
