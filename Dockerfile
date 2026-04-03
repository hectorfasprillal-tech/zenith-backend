FROM python:3.11-slim

WORKDIR /app

# Copiamos solo las dependencias primero
COPY requirements.txt .

# Instalamos dependencias dentro del contenedor (NO del host)
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código (sin venv)
COPY . .

EXPOSE 3000

# Ejecutamos Gunicorn con el Python del contenedor
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "server:app"]

