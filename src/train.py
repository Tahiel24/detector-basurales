import os
import yaml
import torch

from rastervision.core.data import (
    ClassConfig,
    RasterioSourceConfig,
    GeoJSONVectorSourceConfig,
    ClassInferenceTransformerConfig,
    RasterizedSourceConfig,
    RasterizerConfig,
    SemanticSegmentationLabelSourceConfig,
    SceneConfig,
    DatasetConfig,
)

from rastervision.pytorch_learner import (
    SolverConfig,
    SemanticSegmentationLearnerConfig,
    SemanticSegmentationModelConfig,
    SemanticSegmentationGeoDataConfig,
    GeoDataWindowConfig,
    GeoDataWindowMethod,
    Backbone,
)

BACKBONES = {
    "resnet18": Backbone.resnet18,
    "resnet50": Backbone.resnet50,
    "resnet101": Backbone.resnet101,
}


def load_settings(config_path="config/settings.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("--- INICIANDO PIPELINE DE ENTRENAMIENTO ---")

    settings = load_settings("config/settings.yaml")
    model_cfg = settings["model"]
    training_cfg = settings["training"]
    classes_cfg = settings["classes"]

    # 1. Clases
    class_config = ClassConfig(
        names=classes_cfg["names"],
        colors=classes_cfg["colors"],
        null_class=classes_cfg["null_class"],
    )
    class_config.ensure_null_class()
    background_id = class_config.null_class_id
    # id de la clase "basural" = la primera que no es la clase nula
    basural_id = next(
        i for i, n in enumerate(class_config.names) if n != classes_cfg["null_class"]
    )

    # 2. Rutas
    raster_path = "data/raw/imagen.tif"
    vector_path = "data/annotations/basurales.geojson"
    aoi_path = "data/annotations/area_de_interes.geojson"
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/tmp", exist_ok=True)

    # 3. Fuente ráster
    raster_source = RasterioSourceConfig(uris=[raster_path], transformers=[])

    # 4. Etiquetas: GeoJSON (vector) -> rasterizado -> fuente de etiquetas de segmentación
    vector_source = GeoJSONVectorSourceConfig(
        uris=[vector_path],
        ignore_crs_field=True,
        transformers=[ClassInferenceTransformerConfig(default_class_id=basural_id)],
    )
    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(background_class_id=background_id),
        )
    )

    # 5. Escena (el AOI se pasa como lista en aoi_uris)
    scene = SceneConfig(
        id="escena_entrenamiento",
        raster_source=raster_source,
        label_source=label_source,
        aoi_uris=[aoi_path],
    )

    # TEMPORAL: sin escena de validación separada se valida sobre la misma escena
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[scene],
        validation_scenes=[scene],
    )

    # 6. Datos
    window_size = model_cfg["window_size"]
    data_config = SemanticSegmentationGeoDataConfig(
        scene_dataset=dataset,
        window_opts=GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding,
            size=window_size,
            stride=window_size // 2,
        ),
        img_sz=window_size,
        num_workers=2,
    )

    # 7. Solver
    solver_config = SolverConfig(
        batch_sz=training_cfg["batch_size"],
        lr=training_cfg["lr"],
        num_epochs=training_cfg["n_epochs"],
    )

    # 8. Modelo
    backbone_name = model_cfg["architecture"]
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Arquitectura no soportada: {backbone_name}. Usar una de {list(BACKBONES)}"
        )
    model_config = SemanticSegmentationModelConfig(
        backbone=BACKBONES[backbone_name], pretrained=True
    )

    learner_config = SemanticSegmentationLearnerConfig(
        model=model_config,
        data=data_config,
        solver=solver_config,
        output_uri=output_dir,
    )

    # 9. Entrenamiento
    print(f"Construyendo el modelo ({backbone_name})...")
    learner = learner_config.build(tmp_dir="data/tmp")

    print("Comenzando el proceso de entrenamiento...")
    learner.train()

    # 10. Bundle (se guarda en output_uri/model-bundle.zip)
    learner.save_model_bundle()
    print(f"¡Entrenamiento finalizado! Bundle en: {os.path.join(output_dir, 'model-bundle.zip')}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()