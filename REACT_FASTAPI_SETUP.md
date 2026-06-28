# React + FastAPI Quick Start

## Backend (FastAPI)

```powershell
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

## Frontend (React + Vite)

```powershell
cd frontend
npm install
npm run dev
```

Open: http://127.0.0.1:5173

## Notes
- This runs in parallel with your existing Streamlit app.
- API reuses your current agents, dashboard chart logic, and analytics modules.
