# Usamos una imagen oficial de Python como base
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema operativo necesarias para GDAL, Rasterio y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Actualizar pip e instalar herramientas de compilación de Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las librerías especificadas en el requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y las configuraciones al contenedor
COPY config/ /app/config/
COPY src/ /app/src/

# Crear las carpetas de datos y modelos vacías por si no están montadas por volumen
RUN mkdir -p /app/data/raw /app/data/annotations /app/data/outputs /app/models

# Comando por defecto al iniciar el contenedor (se puede sobrescribir al correr docker run)
CMD ["python3"]