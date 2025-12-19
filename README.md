🌿 Crop Disease Detection Web App

A Streamlit-based Crop Disease Detection application that uses a custom Convolutional Neural Network (CNN) to identify crop leaf diseases from images and provide basic treatment advice. The app also supports multilingual output to make it farmer-friendly.

⸻

🚀 Features (Implemented & Accurate)
	•	📸 Upload crop leaf images (JPG / PNG / JPEG)
	•	🧠 Disease prediction using a custom-trained CNN model (PyTorch)
	•	🌱 Supports multiple crops and diseases (Apple, Corn, Grape, Tomato, etc.)
	•	💡 Displays disease-specific treatment advice
	•	🌍 Multilingual UI using Google Translator
	•	English
	•	Hindi
	•	Tamil
	•	Telugu
	•	📷 Optional camera input for live image capture
	•	🖥️ Clean and simple Streamlit interface

⸻

🧠 Model Details
	•	Model Type: Custom CNN (trained from scratch)
	•	Framework: PyTorch
	•	Input Size: 128 × 128 RGB images
	•	Architecture:
	•	3 Convolution layers (32 → 64 → 128 filters)
	•	ReLU activation
	•	MaxPooling after each convolution
	•	Fully connected layer (256 units)
	•	Dropout (0.5) to reduce overfitting
	•	Output layer with 15 classes

⸻

🧩 Supported Outputs
	•	Predicted Disease Name (human-readable)
	•	Treatment / Control Advice for detected disease
	•	Healthy plant confirmation if no disease is detected

⸻

🛠️ Tech Stack
	•	Frontend: Streamlit
	•	Deep Learning: PyTorch
	•	Image Processing: TorchVision, PIL
	•	Language Translation: googletrans
	•	Model Format: .pth (PyTorch state dictionary)

⸻

⚙️ Installation & Setup

1️⃣ Clone the Repository
``` bash
git clone https://github.com/your-username/crop-disease-detector.git
cd crop-disease-detector
```
2️⃣ Create Virtual Environment (Recommended)
``` bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
```
3️⃣ Install Required Libraries
``` bash
pip install streamlit torch torchvision pillow googletrans==4.0.0rc1
```
4️⃣ Place the Trained Model
``` bash
Ensure the trained model file is present in the project root:

crop_disease_cnn.pth
```
5️⃣ Run the Application
``` bash
streamlit run app.py
```

⸻

🧪 How the Application Works
	1.	User uploads or captures a crop leaf image
	2.	Image is resized and normalized
	3.	CNN model predicts the disease class
	4.	Disease label is mapped to a readable name
	5.	Treatment advice is displayed
	6.	Output is translated into the selected language

⸻

🌾 Supported Crops & Diseases (As Implemented)
	•	Apple (Scab, Black Rot, Cedar Rust)
	•	Corn (Gray Leaf Spot, Common Rust, Northern Leaf Blight)
	•	Grape (Black Rot, Esca, Leaf Blight)
	•	Tomato (Early Blight, Late Blight, Leaf Mold, Mosaic Virus, etc.)
	•	Potato, Pepper, Peach, Strawberry, Squash
	•	Healthy class for all crops

⸻

🔐 Limitations (Current Implementation)
	•	Model supports only 15 trained classes
	•	Works on leaf images only
	•	No confidence score displayed
	•	Requires a pre-trained .pth model file

⸻

📈 Future Scope (Not Yet Implemented)
	•	Confidence score visualization
	•	Fertilizer dosage recommendations
	•	Multilingual voice output
	•	Mobile-friendly deployment
	•	Cloud model hosting

⸻

👨‍💻 Author

Lokesh
Student | AI / ML | Deep Learning

⸻

📜 License

This project is intended for academic and educational use.

⸻

Early detection of crop diseases helps farmers reduce losses and improve yield. 🌱
