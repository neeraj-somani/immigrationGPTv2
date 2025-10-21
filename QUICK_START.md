# ImmigrationGPT - Quick Start Guide

## 🚀 Easy Startup Options

### Option 1: Start Everything at Once (Recommended)
**Windows:**
```bash
start_all.bat
```

**Linux/Mac:**
```bash
./start_all.sh
```

### Option 2: Start Applications Separately

**Backend Only:**
- Windows: `start_backend.bat`
- Linux/Mac: `./start_backend.sh`

**Frontend Only:**
- Windows: `start_frontend.bat` 
- Linux/Mac: `./start_frontend.sh`

### Option 3: Manual Commands

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Access Points

- **Frontend (Chat Interface):** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## ⚙️ Configuration

1. **API Keys Setup:**
   - Open the frontend at http://localhost:3000
   - Click the settings icon (⚙️)
   - Add your API keys:
     - OpenAI API Key (required)
     - Tavily API Key (required for web search)
     - LangChain API Key (optional, for tracing)

2. **Environment Variables:**
   - Copy `env.template` to `.env` in the root directory
   - Add your API keys to the `.env` file

## 🔧 Troubleshooting

### Backend Issues:
- Ensure Python 3.8+ is installed
- Check that all dependencies are installed: `pip install -r backend/requirements.txt`
- Verify API keys are set correctly

### Frontend Issues:
- Ensure Node.js 16+ is installed
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check that port 3000 is available

### Port Conflicts:
- Backend runs on port 8000
- Frontend runs on port 3000
- Change ports in the respective configuration files if needed

## 📝 Features

- **AI-Powered Immigration Assistant** with RAG capabilities
- **Real-time Web Search** for current information
- **Interactive Chat Interface** with markdown support
- **Settings Management** for API keys
- **Health Monitoring** and error handling

## 🛠️ Development

- Backend: FastAPI with Python
- Frontend: Next.js with React and TypeScript
- Styling: Tailwind CSS
- State Management: Zustand
- HTTP Client: Axios
