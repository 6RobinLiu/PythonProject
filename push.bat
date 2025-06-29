@echo off
set PROJECT_DIR=C:\PythonProject


set REPO_SSH=git@github.com:6RobinLiu/PythonProject.git

cd /d %PROJECT_DIR%

echo ============================
echo [1] Checking if git repo is initialized...
echo ============================
if not exist ".git" (
    echo Not a git repository. Initializing...
    git init
    git remote add origin %REPO_SSH%
) else (
    echo Git repository already exists.
)

echo.
echo ============================
echo [2] Checking Git Status...
echo ============================
git status
echo.

echo ============================
echo [3] Adding changes...
echo ============================
git add .
echo.

echo ============================
echo [4] Committing...
echo ============================
git commit -m "Auto update"
echo.

echo ============================
echo [5] Pulling from GitHub (rebase)...
echo ============================
git pull origin main --rebase
echo.

echo ============================
echo [6] Pushing to GitHub...
echo ============================
git push origin main
echo.

echo ============================
echo Done.
echo ============================
pause
