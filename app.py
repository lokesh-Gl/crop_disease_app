import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from googletrans import Translator
import streamlit as st

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

# --> Multilingual Translation <--

translator = Translator()

def translate_text(text, dest):
    try:
        return translator.translate(text, dest=dest).text
    except:
        return text

# --> Disease Name Mapping and Advice <--

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
    "Corn Gray Leaf Spot": "Use resistant varieties and apply fungicide like Azoxystrobin when disease appears.",
    "Tomato Late Blight": "Apply copper-based fungicide. Remove and destroy infected plants.",
    "Healthy": "Your plant is healthy! Maintain good watering practices, monitor regularly, and follow good field hygiene."
}

# --> Streamlit App <--

def main():
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
    img = img.convert("RGB")  # Ensure compatibility
    
    st.image(img, caption=translate_text("Uploaded Image", lang_code), use_container_width=True)

    # Convert image to NumPy array safely
    img_np = np.array(img, dtype=np.uint8)

    # Convert back to PIL Image before transformations
    img_pil = Image.fromarray(img_np)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    input_tensor = transform(img_pil).unsqueeze(0)  # Apply transformations

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        class_names = list(disease_map.keys())
        predicted_label = disease_map.get(class_names[pred_class], "Unknown Disease")

    st.success(translate_text(f"🩺 Predicted Disease: {predicted_label}", lang_code))
    st.info(translate_text(f"Treatment Advice: {advice_map.get(predicted_label, 'No specific advice available.')}", lang_code))

if __name__ == "__main__":
    main()
