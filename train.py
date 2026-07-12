from ultralytics import YOLO
import torch

def main():
    # -----------------------------
    # Check device
    # -----------------------------
    device = "0"

    print("Using device:", device)
    print("Torch CUDA available:", torch.cuda.is_available())

    # -----------------------------
    # Load pretrained model
    # -----------------------------
    model = YOLO("yolo11n.pt")   # nano model 

    # -----------------------------
    # Train on GPU
    # -----------------------------
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        cache=True,
        pretrained=True,
        optimizer="SGD",
        verbose=True
    )

if __name__ == "__main__":
    main()