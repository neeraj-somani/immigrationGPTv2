#!/bin/bash
echo "Starting ImmigrationGPT Full Stack Application..."
echo

echo "[1/3] Installing Backend Dependencies..."
cd backend
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install backend dependencies"
    exit 1
fi

echo
echo "[2/3] Installing Frontend Dependencies..."
cd ../frontend
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install frontend dependencies"
    exit 1
fi

echo
echo "[3/3] Starting Applications..."
echo "Starting Backend on http://localhost:8000"
cd ../backend
python main.py &
BACKEND_PID=$!

echo "Waiting 3 seconds for backend to initialize..."
sleep 3

echo "Starting Frontend on http://localhost:3000"
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo
echo "✅ Both applications are starting!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo
echo "Press Ctrl+C to stop both applications"

# Wait for user interrupt
trap "echo 'Stopping applications...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
