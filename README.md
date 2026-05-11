# 🚗 SafeDrive AI – Version 1.0
### AI-Powered Driver Safety & Monitoring System

SafeDrive AI is an intelligent driver-monitoring system that uses computer vision and machine learning to detect **drowsiness, distraction, alcohol influence, and unsafe driving patterns** in real time.  
Its goal is to enhance road safety and prevent avoidable accidents through early warnings and instant feedback.

---

## ✨ Key Features

- 😴 **Drowsiness Detection**  
  Identifies eye-closure duration, blinking patterns, and yawning.

- 👀 **Distraction Detection**  
  Detects when the driver looks away or shows dangerous inattention.

- 🍺 **Alcohol Influence Warning**  
  Uses trained classification models to identify alcohol-related facial indicators.

- 📡 **Real-Time Video Processing**  
  Processes webcam/video frames with low latency.

- 🔔 **Smart Alert System**  
  Sends audio + visual warnings when unsafe behavior is detected.

---

## 🧠 Tech Stack

- **Python**
- **OpenCV**
- **TensorFlow / Keras**
- **Deep Learning Models (Classification + CNN)**
- **Custom Training Datasets**

---

## 📁 Project Structure (Simplified)


SafeDrive_AI-Version-1.0/
│
├── models/ # Trained ML models
├── datasets/ # Training datasets (alcohol, drowsiness, etc.)
├── core/ # Detection logic (eyes, face, pose)
├── interface/ # UI files (HTML/CSS/JS if a web dashboard exists)
├── utils/ # Helper scripts (preprocessing, alerts, etc.)
└── main.py # Main entry script


---




Camera Frame
     ↓
YOLOv8n (Person Detection)
     ↓
Is person detected?
     ├── NO → "No person detected"
     │
     └── YES
          ↓
   Crop face / ROI
          ↓
 Alcohol Classifier Model
          ↓
   Alcoholic / Non-Alcoholic

   

## 🚀 How to Run the Project

1. **Install required libraries**
   ```bash
   pip install -r requirements.txt

   Start the application


   python main.py

   Allow webcam access
The system will automatically begin monitoring.


🙌 Contribution Guidelines

Pull requests are welcome!
Improve models, UI, documentation, or add new safety features.

