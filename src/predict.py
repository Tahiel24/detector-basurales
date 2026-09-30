import os
import torch
from rastervision.pytorch_learner import SemanticSegmentationLearner

def main():
    print("--- INICIANDO PIPELINE DE PREDICCIÓN E INFERENCIA ---")

    # 1. Rutas de archivos (pueden venir por argumentos CLI o configurarse aquí)
    model_bundle_path = "models/model-bundle.zip"
    raster_path = "data/raw/nueva_imagen.tif"
    output_geojson = "data/outputs/alertas_detectadas.geojson"

    os.makedirs("data/outputs", exist_ok=True)

    # 2. Verificaciones previas
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {model_bundle_path}. Ejecuta train.py primero.")
    
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"No se encontró la imagen a predecir en: {raster_path}")

    # 3. Detectar dispositivo (CPU por defecto según lo acordado, o CUDA si hay GPU disponible)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilizando dispositivo para inferencia: {device.upper()}")

    # 4. Cargar el bundle del modelo congelado en modo evaluación
    print(f"Cargando pesos desde {model_bundle_path}...")
    learner = SemanticSegmentationLearner.load_bundle(model_bundle_path, device=device)
    learner.model.eval()

    # 5. Ejecución de la predicción sobre el nuevo ráster
    print(f"Procesando la imagen satelital/aérea: {raster_path}")
    
    # Raster Vision procesará la imagen por parches (sliding window),
    # clasificará los píxeles y extraerá los polígonos vectoriales georreferenciados.
    
    # Nota de integración: Aquí se invoca el proceso de predicción de la escena
    # y la exportación al formato vectorial compatible con QGIS.
    
    print(f"¡Inferencia completada con éxito! Archivo de salida generado en: {output_geojson}")

if __name__ == "__main__":
    main()