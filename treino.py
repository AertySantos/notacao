from ultralytics import YOLO

arquivo_config = "dataset_/data.yaml"

model = YOLO("yolo11m.pt")  # ou 'yolov8x.yaml' se preferir do zero

resultados = model.train(
    data=arquivo_config,
    epochs=300,
    imgsz=2048,
    rect=True,
    name='yolov8x_finetune_lines',
    device='cuda:1',
    batch=8,
    degrees=3,
    translate=0.05,
    scale=0.4,
    shear=0.5,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.1,
    mosaic=0.3,
    mixup=0.05,
    conf=0.4,
    iou=0.5
)
