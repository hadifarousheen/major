cd backend
python -m venv venv
source venv/Scripts/activate 
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload

cd frontend
python -m http.server 5500


http://127.0.0.1:5500/index.html
