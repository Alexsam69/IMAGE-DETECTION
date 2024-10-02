import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Define vehicle classes (replace with actual class IDs from your model)
vehicle_classes = [2, 3, 5]

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Function to display processed image with bounding boxes
def display_processed_image(image_bytes, results, model):
    preprocessed_image = preprocess_image(cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))

    for box, score, label_id in zip(results['boxes'], results['scores'], results['labels']):
        if label_id.item() in vehicle_classes:
            x_min, y_min, x_max, y_max = box.tolist()
            cv2.rectangle(preprocessed_image, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # Optionally, add label text
            label = f"{model.module.names[label_id.item()]} - {score:.2f}"
            cv2.putText(preprocessed_image, label, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Convert processed image back to RGB and display in Streamlit
    rgb_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, channels="RGB", use_column_width=True)

# Function to detect vehicles
def detect_vehicles(image_bytes, model):
    image = preprocess_image(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))
    with torch.no_grad():
        results = model(image)[0]
    return results

# Main app function
def main():
    st.title("Vehicle Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        results = detect_vehicles(image_bytes, model)
        display_processed_image(image_bytes, results, model)

if __name__ == "__main__":
    main()

        st.success(f"Detected {vehicle_count} vehicles!")
        display_processed_image(image_bytes, results, model)
