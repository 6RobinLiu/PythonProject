@echo off
cd /d C:\PythonProject

echo --- Git 状态检查 ---
git status
echo.

echo --- 添加更改文件 ---
git add .

echo --- 本地提交 ---
git commit -m "Auto update"

echo --- 拉取远程分支（防止冲突） ---
git pull origin main --rebase

echo --- 推送到 GitHub ---
git push origin main

echo.
echo --- 已完成推送 ---
pause
