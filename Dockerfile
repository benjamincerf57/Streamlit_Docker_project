FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances 
RUN pip install -r requirements.txt

EXPOSE 8501

# Lancer Streamlit lors du démarrage du conteneur
CMD ["streamlit", "run", "app.py"]
