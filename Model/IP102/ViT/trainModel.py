# from ultralytics import YOLO
# import torch

# if __name__ == "__main__":
#     print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
#     print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

#     # Load a pre-trained YOLOv8 model
#     model = YOLO("yolov8n.pt")  

#     # Train the model on your dataset
#     model.train(data="plant_yolo.yaml", epochs=50, imgsz=640, batch=8, device="cuda")

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # Load a YOLOv8 model (small or medium depending on VRAM)
    model = YOLO("yolov8m.pt")  # 's' for small, 'n' for nano, 'm' for medium

    # Train the model on your dataset without saving intermediate checkpoints
    model.train(
        data="plant_yolo.yaml",   # your dataset config
        epochs=140,               # training epochs
        imgsz=640,                # image size
        batch=16,                 # batch size
        device=device,
        lr0=0.001,                # initial learning rate
        augment=True,             # standard augmentation
        patience=10,              # early stopping patience
        save_period=-1,           # disables saving checkpoints during training
        exist_ok=True,             # overwrite if run folder exists
        workers=2
    )

    print("Training started successfully.")
