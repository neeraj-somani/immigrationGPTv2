@echo off
echo Starting ImmigrationGPT Full Stack Application...
echo.

echo [1/3] Installing Backend Dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Installing Frontend Dependencies...
cd ..\frontend
npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Applications...
echo Starting Backend on http://localhost:8000
start "ImmigrationGPT Backend" cmd /k "cd ..\backend && python main.py"

echo Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak >nul

echo Starting Frontend on http://localhost:3000
start "ImmigrationGPT Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Both applications are starting!
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window...
pause >nul
