@echo off
cd /d C:\PythonProject

echo ============================
echo Checking Git Status...
echo ============================
git status
echo.

echo ============================
echo Adding all changes...
echo ============================
git add .
echo.

echo ============================
echo Committing changes...
echo ============================
git commit -m "Auto update"
echo.

echo ============================
echo Pulling latest changes from GitHub...
echo ============================
git pull origin main --rebase
echo.

echo ============================
echo Pushing to GitHub...
echo ============================
git push origin main
echo.

echo ============================
echo Push completed successfully!
echo ============================
pause
