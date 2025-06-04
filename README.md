🍌 Automated Fruit Quality Analysis Using Jetson Nano
A real-time fruit quality assessment system built on the NVIDIA Jetson Nano platform, leveraging computer vision and deep learning to classify and evaluate the freshness, ripeness, or defects in fruits (currently bananas). Optimized for edge AI deployment, this project uses a camera module to capture images and a custom-trained MobileNetV2 model for classification.




📁 Project Structure
Automated-Fruit-Quality-Analysis-Using-Jetson-Nano/
│
├── model/                      # Trained MobileNetV2 model (TensorRT optimized)
├── images/                     # Sample input/output images
├── dataset/                    # Dataset used for training (if included)
├── src/
│   ├── classify.py             # Inference script for Jetson Nano
│   ├── camera_capture.py       # Captures images using CSI/USB camera
│   └── utils.py                # Helper functions for preprocessing
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
└── LICENSE


🚀 Features
Real-time classification of banana quality: Good, Bad, Intermediate

Built using MobileNetV2 trained on a custom dataset

Optimized for Jetson Nano using TensorRT

Easy integration with conveyor belt systems

Designed for low-latency and on-device inference

🧠 Model Details
Model: MobileNetV2

Framework: PyTorch → TensorRT

Accuracy: 93% on validation dataset

Dataset Size: 6.5k+ images per class for training, 650+ images per class for validation

🛠️ Setup Instructions-
# Clone the repository
git clone https://github.com/your-username/Automated-Fruit-Quality-Analysis-Using-Jetson-Nano.git
cd Automated-Fruit-Quality-Analysis-Using-Jetson-Nano

# Install dependencies
pip install -r requirements.txt

# Run the classification
python src/classify.py

📸 Project in Action
Jetson Nano + Camera + Conveyor Setup	*Predicted: Good 🍌

🔍 Real-time Classification Example
Input: Live banana image captured via camera

Model: MobileNetV2 (TensorRT-optimized)

Output Classes: Good, Intermediate, Bad

FPS: ~18–22 on Jetson Nano (with TensorRT)

Accuracy Achieved: ~93% on validation data
images/img1.jpg
images/ing2.png


