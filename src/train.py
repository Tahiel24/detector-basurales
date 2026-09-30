import os
import yaml
import torch

from rastervision.core.data import (
    ClassConfig,
    RasterioSourceConfig,
    GeoJSONSourceConfig,
    SceneConfig,
    DatasetConfig
)

from rastervision.pytorch_learner import (
    SolverConfig,
    SemanticSegmentationLearnerConfig,
    SemanticSegmentationGeoDataConfig,
    Backbone
)

def load_settings(config_path="config/settings.yaml"):
    """Carga los parámetros globales desde el archivo YAML externo."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    return settings

def main():
    print("--- INICIANDO PIPELINE DE ENTRENAMIENTO CON SETTINGS.YAML ---")

    # 1. Cargar la configuración externa
    settings = load_settings("config/settings.yaml")
    
    model_cfg = settings.get("model", {})
    training_cfg = settings.get("training", {})
    classes_cfg = settings.get("classes", {})

    # 2. Configuración de Clases dinámicas desde el YAML
    class_config = ClassConfig(
        names=classes_cfg.get("names", ["background", "basural"]),
        colors=classes_cfg.get("colors", ["lightgray", "darkred"]),
        null_class=classes_cfg.get("null_class", "background")
    )
    class_config.ensure_null_class()

    # 3. Rutas a los datos locales
    raster_path = "data/raw/imagen_prueba.tif"
    vector_path = "data/annotations/etiquetas.geojson"
    aoi_path = "data/annotations/aoi.geojson"  
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Configuración de fuentes de datos ráster y etiquetas vectoriales
    raster_source = RasterioSourceConfig(
        uris=[raster_path],
        transformers=[]
    )
    
    vector_source = GeoJSONSourceConfig(
        uris=[vector_path],
        default_class_id=1
    )

    scene = SceneConfig(
        id="escena_entrenamiento",
        raster_source=raster_source,
        label_source=vector_source,
        aoi_polygons_uri=aoi_path  # Delimitación espacial del área útil
    )

    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[scene],
        validation_scenes=[]
    )

    # 5. Parámetros leídos del YAML para Dataset y Solver
    window_size = model_cfg.get("window_size", 480)
    
    data_config = SemanticSegmentationGeoDataConfig(
        dataset=dataset,
        window_size=window_size,
        stride=window_size // 2,
        num_workers=2
    )

    solver_config = SolverConfig(
        batch_size=training_cfg.get("batch_size", 4),
        lr=training_cfg.get("lr", 0.01),
        num_epochs=training_cfg.get("n_epochs", 15),
        mixed_precision=training_cfg.get("mixed_precision", False)
    )

    # Seleccionar arquitectura basada en el YAML (por defecto resnet18)
    backbone_name = model_cfg.get("architecture", "resnet18")
    backbone = Backbone.resnet18 if backbone_name == "resnet18" else Backbone.resnet50

    learner_config = SemanticSegmentationLearnerConfig(
        backend=backbone,
        data=data_config,
        solver=solver_config,
        output_dir=output_dir,
        model_weights=None
    )

    # 6. Construcción y Ejecución del Entrenamiento
    print(f"Construyendo el modelo ({backbone_name}) y preparando el dataset...")
    learner = learner_config.build(tmp_dir="data/tmp")
    
    print("Comenzando el proceso de entrenamiento...")
    learner.train()

    # 7. Guardado del Model Bundle resultante
    bundle_path = os.path.join(output_dir, "model-bundle.zip")
    learner.save_bundle(bundle_path)
    print(f"¡Entrenamiento finalizado con éxito! Bundle guardado en: {bundle_path}")

    # Limpieza de memoria
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()