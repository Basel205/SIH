# import streamlit as st
# from ultralytics import YOLO
# import cv2
# from PIL import Image
# import numpy as np
# import os
# import glob

# # -------------------------------
# # Automatically find the latest training folder
# base_weights_path = "runs/detect"
# train_folders = sorted(glob.glob(os.path.join(base_weights_path, "train*")), key=os.path.getmtime, reverse=True)

# if not train_folders:
#     st.error("No trained model found. Please train the model first.")
#     st.stop()

# latest_train_folder = train_folders[0]
# weights_path = os.path.join(latest_train_folder, "weights", "best.pt")

# # Load the trained YOLOv8 model
# model = YOLO(weights_path)

# # -------------------------------
# st.title("PlantYOLO - Plant Disease Detection")
# st.write("Upload a leaf image and see the detected disease class!")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     # Convert uploaded file to OpenCV image
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
    
#     # Run inference
#     results = model.predict(image_np, conf=0.25, save=False)  # adjust confidence as needed
    
#     # Draw boxes on the image
#     result_img = results[0].plot()
    
#     # Show result in Streamlit
#     st.image(result_img, caption="Prediction", use_column_width=True)


import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from pytorch_model import InsectModel  # your model definition

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InsectModel(num_classes=102)
model.load_state_dict(torch.load('vit_best.pth', map_location=device))
model.to(device)
model.eval()

# Load classes
with open('classes.txt') as f:
    classes = [line.strip().split(' ', 1)[1] for line in f]

st.title("Insect Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.softmax(1).argmax(1).item()
        st.write(f"Predicted class: {classes[pred]}")
