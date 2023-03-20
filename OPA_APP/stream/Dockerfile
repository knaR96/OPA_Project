#  image existante de Python 
FROM python:3.9-slim-buster


# Définit le répertoire de travail comme /app
WORKDIR /stream
# Copier les fichiers locaux dans le conteneur
COPY . .

# Installe les dépendances de l'API depuis requirements.txt
RUN pip install -r requirements.txt

# Ajoute le volume pour stocker les données de la base de données
#VOLUME /data

# Définit la commande à exécuter lorsque le conteneur est lancé
CMD ["python", "stream.py"]