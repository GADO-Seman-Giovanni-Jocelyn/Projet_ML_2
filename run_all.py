import subprocess
import time
import webbrowser
import sys


# Lancer l'API FastAPI sur http://localhost:8000
api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "API:app", "--reload"])
# Attendre un petit moment que l'API démarre
time.sleep(2)

streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"])
# (Optionnel) ouvrir automatiquement le dashboard
webbrowser.open("http://localhost:8501")

# Attendre que les deux processus tournent
try:
    api_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    print("Arrêt manuel détecté. Fermeture des serveurs...")
    api_process.terminate()
    streamlit_process.terminate()
