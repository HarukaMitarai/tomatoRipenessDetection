import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from typing import List
from pydantic import BaseModel


# Load the YOLO model
model = YOLO("laboloTomato_yolov8s_best.pt")

st.title("Tomato Ripeness Prediction")

# Define a data class for predictions
class Prediction(BaseModel):
    class_name: str
    confidence: float
    bounding_box: List[int]

# Function to perform object detection and annotate the image
def predict_and_process(image):
    """Performs object detection and returns predictions and the processed image."""
    results = model(image)  # Run YOLOv8 inference
    predictions = []

    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls.cpu().numpy())]
            confidence = box.conf.cpu().item()
            bounding_box = box.xyxy.cpu().numpy()[0].astype(int).tolist()
            predictions.append(Prediction(class_name=class_name, confidence=confidence, bounding_box=bounding_box))

            # Draw bounding box and label on the image
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
    
    return predictions, image

# Main Streamlit application logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Perform object detection and get predictions
        predictions, processed_image = predict_and_process(image)
        
        # streamlit によってブラウザに描画される画像の色をBGRからRGBに変換（画像の青みを解消するため）
        processed_image_disp = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Display the processed image
        st.image(processed_image_disp, caption="Processed Image", use_container_width=True)

        # Display the predictions
        st.subheader("Predictions:")
        for prediction in predictions:
            st.write(f"- Class: {prediction.class_name}, Confidence: {prediction.confidence:.2f}, Bounding Box: {prediction.bounding_box}")

        # Create a temporary file to save the processed image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_filepath = temp_file.name
            cv2.imwrite(temp_filepath, processed_image)
            
            # Provide a download link for the processed image
            with open(temp_filepath, "rb") as f:
                st.download_button("Download Processed Image", data=f, file_name="processed_image.jpg")  

        # Remove the temporary file after download
        os.remove(temp_filepath)

    except Exception as e:
        st.error(f"Error processing image: {e}")