import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from googletrans import Translator

# --> Load Custom CNN Architecture <--

class CropDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(CropDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 15)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#  -->  Multilingual Translation  <--

translator = Translator()

def translate_text(text, dest):
    try:
        return translator.translate(text, dest=dest).text
    except:
        return text


#  --> Disease Name Mapping and Advice <--

disease_map = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Apple Black Rot",
    "Apple___Cedar_apple_rust": "Apple Cedar Rust",
    "Apple___Healthy": "Healthy",

    "Blueberry___Healthy": "Healthy",

    "Cherry_(including_sour)___Powdery_mildew": "Cherry Powdery Mildew",
    "Cherry_(including_sour)___Healthy": "Healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
    "Corn_(maize)___Healthy": "Healthy",

    "Grape___Black_rot": "Grape Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight (Isariopsis Leaf Spot)",
    "Grape___Healthy": "Healthy",

    "Orange___Haunglongbing_(Citrus_greening)": "Citrus Greening (Huanglongbing)",

    "Peach___Bacterial_spot": "Peach Bacterial Spot",
    "Peach___Healthy": "Healthy",

    "Pepper__bell___Bacterial_spot": "Bell Pepper Bacterial Spot",
    "Pepper__bell___Healthy": "Healthy",

    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___Healthy": "Healthy",

    "Raspberry___Healthy": "Healthy",

    "Soybean___Healthy": "Healthy",

    "Squash___Powdery_mildew": "Squash Powdery Mildew",

    "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch",
    "Strawberry___Healthy": "Healthy",

    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mite Infestation",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___Healthy": "Healthy",
    "Tomato___Tomato_Leaf_Curl_Virus": "Tomato Leaf Curl Virus"
}

advice_map = {
    "Apple Scab": "Apply fungicides like captan or myclobutanil early in the season. Remove fallen leaves and infected fruits.",
    "Apple Black Rot": "Prune infected branches and apply fungicides. Remove mummified fruits from the tree.",
    "Apple Cedar Rust": "Remove nearby juniper plants. Apply fungicides during the early growing season.",
    "Cherry Powdery Mildew": "Apply sulfur-based fungicide. Improve airflow by proper pruning.",
    "Corn Gray Leaf Spot": "Use resistant varieties and apply fungicide like Azoxystrobin when disease appears.",
    "Corn Common Rust": "Apply fungicides such as mancozeb or propiconazole. Remove affected leaves.",
    "Corn Northern Leaf Blight": "Use disease-resistant hybrids and apply fungicides early in the season.",
    "Grape Black Rot": "Remove infected leaves and apply fungicide like captan or mancozeb at early bloom.",
    "Grape Esca (Black Measles)": "Prune and remove infected vines. There's no effective chemical control; maintain vine vigor.",
    "Grape Leaf Blight (Isariopsis Leaf Spot)": "Apply protective fungicides and remove infected leaves and debris.",
    "Citrus Greening (Huanglongbing)": "Remove infected trees. Control psyllid vectors using insecticides.",
    "Peach Bacterial Spot": "Apply copper-based bactericides. Use resistant cultivars and avoid overhead irrigation.",
    "Bell Pepper Bacterial Spot": "Spray copper fungicides. Use pathogen-free seeds and rotate crops.",
    "Potato Early Blight": "Apply fungicides like chlorothalonil and rotate crops annually.",
    "Potato Late Blight": "Use fungicides like metalaxyl and destroy infected plant material.",
    "Squash Powdery Mildew": "Apply fungicides like sulfur or potassium bicarbonate. Ensure good airflow.",
    "Strawberry Leaf Scorch": "Use resistant cultivars and apply fungicide if necessary. Avoid overhead irrigation.",
    "Tomato Early Blight": "Use a fungicide like Mancozeb or Chlorothalonil. Rotate crops and avoid overhead irrigation.",
    "Tomato Late Blight": "Apply copper-based fungicide. Remove and destroy infected plants.",
    "Tomato Leaf Mold": "Improve ventilation and apply fungicides like Chlorothalonil or copper-based treatments.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves and apply fungicide like mancozeb weekly.",
    "Tomato Spider Mite Infestation": "Spray neem oil or insecticidal soap. Increase humidity to reduce mites.",
    "Tomato Target Spot": "Use fungicide with chlorothalonil. Improve air circulation and avoid moisture on leaves.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies with insecticide. Remove and destroy infected plants.",
    "Tomato Mosaic Virus": "Disinfect tools and hands. Avoid smoking near tomato plants.",
    "Tomato Bacterial Spot": "Use copper fungicide weekly. Remove and destroy affected plants.",
    "Tomato Leaf Curl Virus": "Control whitefly vectors and use virus-resistant seed varieties.",
    "Healthy": "Your plant is healthy! Maintain good watering practices, monitor regularly, and follow good field hygiene."
}


# --> Streamlit App  <--
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")
langs = {"English": "en", "Hindi (हिंदी)": "hi", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te"}
lang_ui = st.selectbox("Choose Language", list(langs.keys()))
lang_code = langs.get(lang_ui, "en")


st.title(translate_text("Crop Disease Detection", lang_code))
st.write(translate_text("Upload or capture an image of a crop leaf to detect disease and get fertilizer suggestions.", lang_code))

@st.cache_resource
def load_model():
    model = CropDiseaseCNN(num_classes=15)
    model.load_state_dict(torch.load("crop_disease_cnn.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

image_file = st.file_uploader(translate_text("📁 Upload an Image", lang_code), type=["jpg", "png", "jpeg"])
img = None

st.write("---")
use_camera = st.checkbox(translate_text("📷 Use Camera Instead", lang_code))

if use_camera:
    if st.button(translate_text("Click to Open Camera", lang_code)):
        camera_image = st.camera_input(translate_text("📸 Take a Photo", lang_code))
        if camera_image:
            img = Image.open(camera_image).convert("RGB")
else:
    if image_file:
        img = Image.open(image_file).convert("RGB")

if img:
    # Ensure image is in RGB format
    img = Image.open(img).convert('RGB')

    st.image(img, caption=translate_text("Uploaded Image", lang_code), use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # For RGB images
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        class_names = list(disease_map.keys())
        predicted_key = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
        predicted_label = disease_map.get(predicted_key, "Unknown Disease")

    st.success(translate_text(f"🩺 Predicted Disease: {predicted_label}", lang_code))

    if predicted_label != "Healthy":
        advice = advice_map.get(predicted_label, "No specific advice available.")
        st.info(translate_text(f"Treatment Advice: {advice}", lang_code))
    else:
        st.info(translate_text("Your crop appears healthy. Keep practicing good field hygiene, crop rotation, and regular pest inspection.", lang_code))

