cd backend
python -m venv venv
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload

cd frontend
python -m http.server 5500
