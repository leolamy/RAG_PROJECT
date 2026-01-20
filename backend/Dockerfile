FROM python:3.11

WORKDIR /app

# outils système nécessaires
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copie du code source
COPY . .
RUN mkdir -p data

# Lancement du serveur API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]