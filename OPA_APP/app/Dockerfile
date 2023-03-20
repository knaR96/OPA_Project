# Utilise l'image Python 3.8
FROM python:3.8

# Copie le contenu du répertoire courant vers /app
COPY . /app

# Définit le répertoire de travail comme /app
WORKDIR /app

# Installe les dépendances de l'API depuis requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#fcopie le fichier de configuration de streamlit
COPY .streamlit/config.toml /root/.streamlit/config.toml
# Expose le port 5000 pour que l'API soit accessible
EXPOSE 8501

# Définit la commande à exécuter lorsque le conteneur est lancé
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]