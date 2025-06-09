import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from googletrans import Translator

# ------------------------
# Define CNN model
# ------------------------
class CropDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(CropDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------
# Translation Function
# ------------------------
translator = Translator()
def translate_text(text, dest_language='en'):
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception:
        return text

# ------------------------
# Label & Advice Maps
# ------------------------
disease_map = {
    'Apple___Black_rot': 'Apple - Black rot',
    'Apple___healthy': 'Apple - Healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Corn - Cercospora leaf spot',
    'Corn_(maize)___Common_rust_': 'Corn - Common rust',
    'Corn_(maize)___healthy': 'Corn - Healthy',
    'Grape___Black_rot': 'Grape - Black rot',
    'Grape___healthy': 'Grape - Healthy',
    'Potato___Early_blight': 'Potato - Early blight',
    'Potato___healthy': 'Potato - Healthy',
    'Potato___Late_blight': 'Potato - Late blight',
    'Tomato___Bacterial_spot': 'Tomato - Bacterial spot',
    'Tomato___Early_blight': 'Tomato - Early blight',
    'Tomato___Late_blight': 'Tomato - Late blight',
    'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato___healthy': 'Tomato - Healthy',
}

advice_map = {
    "Apple - Black rot": "Use fungicides like Captan or Mancozeb. Remove infected fruit.",
    "Corn - Cercospora leaf spot": "Rotate crops and use resistant hybrids.",
    "Corn - Common rust": "Use resistant varieties. Apply fungicides like Mancozeb.",
    "Grape - Black rot": "Apply fungicides. Prune infected areas and maintain airflow.",
    "Potato - Early blight": "Use certified seeds. Apply fungicides like Chlorothalonil.",
    "Potato - Late blight": "Spray fungicides such as Metalaxyl. Ensure proper drainage.",
    "Tomato - Bacterial spot": "Avoid overhead irrigation. Use copper-based fungicides.",
    "Tomato - Early blight": "Practice crop rotation. Use fungicides like Azoxystrobin.",
    "Tomato - Late blight": "Apply protective fungicides. Remove infected plants.",
    "Tomato - Leaf Mold": "Use resistant cultivars. Ensure good air circulation.",
}

# ------------------------
# Streamlit UI Setup
# ------------------------
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
languages = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
}
language_ui = st.selectbox("Choose Language", list(languages.keys()))
language_code = languages[language_ui]

st.title(translate_text("Crop Disease Detection", language_code))
st.write(translate_text("Upload or capture a crop leaf image to detect diseases.", language_code))

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    model = CropDiseaseCNN(num_classes=15)
    model.load_state_dict(torch.load("crop_disease_cnn.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ------------------------
# Image Input
# ------------------------
image_file = st.file_uploader(translate_text("Upload an Image", language_code), type=["jpg", "png", "jpeg"])
use_camera = st.checkbox(translate_text("Use Camera Instead", language_code))

image = None
if use_camera:
    camera_image = st.camera_input(translate_text("Take a Photo", language_code))
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
elif image_file is not None:
    image = Image.open(image_file).convert("RGB")

# ------------------------
# Image Processing & Prediction
# ------------------------
if image is not None:
    st.image(image, caption=translate_text("Uploaded Image", language_code), use_column_width=True)
    try:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()

        class_names = list(disease_map.keys())
        class_name = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"
        disease_label = disease_map.get(class_name, "Unknown Disease")
        
        st.success(translate_text(f"Predicted Disease: {disease_label}", language_code))

        if "healthy" in disease_label.lower():
            st.info(translate_text("Your crop appears healthy. Keep practicing good field hygiene and regular checks.", language_code))
        else:
            treatment_advice = advice_map.get(disease_label, "No specific advice available.")
            st.info(translate_text(f"Treatment Advice: {treatment_advice}", language_code))

    except Exception as e:
        st.error(translate_text("Error during prediction. Please ensure a valid crop leaf image.", language_code))
