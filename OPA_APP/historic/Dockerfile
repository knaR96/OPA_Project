#  image existante de Python 
FROM python:3.9-slim-buster


# Définit le répertoire de travail comme /app
WORKDIR /historic
# Copier les fichiers locaux dans le conteneur
COPY . .
#lances de l'API depuis requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Définit la commande à exécuter lorsque le conteneur est lancé
CMD ["python", "historic.py"]