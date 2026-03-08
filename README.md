# 🐟🐠 Fish Species and Disease Detection using Deep Learning 🧠🦠

> **An End-to-End Computer Vision system that detects fish species and identifies diseases from fish images using Deep Learning.**

---

## 📌 Project Description

This project leverages two specialized AI models to automate fish health monitoring:
1. **🐟 Fish Species Classification Model**
2. **🦠 Fish Disease Detection Model**

This system is designed to provide scalable, AI-based monitoring solutions for:
* 🌊 **Aquaculture farms**
* 🎣 **Fisheries monitoring**
* 🔬 **Marine research**
* ⚙️ **Smart fish farming systems**

---

## 🎯 Project Objectives

- [x] **Automatically identify** fish species from images.
- [x] **Detect diseases** affecting fish with high accuracy.
- [x] **Reduce manual inspection** time and human error.
- [x] **Assist aquaculture monitoring** for better yield and health.
- [x] **Build a scalable** AI-based system for real-world deployment.

---

## 🧠 System Architecture

```text
                    📷 Input Image
                         │
                         ▼
                🧹 Image Preprocessing
                         │
                         ▼
        🐟 Fish Species Classification Model
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                │
         Predicted Species        │
                                  ▼
                   🦠 Fish Disease Detection Model
                                  │
                                  ▼
                         Predicted Disease
```

---

## 🤖 Models Used

### 🐟 1. Fish Species Classification Model
* **Model Type:** CNN / ResNet-Based Classifier
* **Purpose:** Identify the exact species of fish from a given image.
* **Reasoning:** Convolutional Neural Networks (CNNs) excel at image classification by automatically learning spatial hierarchies of visual patterns from raw pixel data.

### 🦠 2. Fish Disease Detection Model
* **Model Type:** EfficientNet / CNN-Based Classifier
* **Purpose:** Detect and classify diseases affecting the identified fish.
* **Possible Disease Classes:**
  * 🟢 Healthy
  * ⚪ White Spot Disease
  * 🔴 Red Spot Disease
  * ⚫ Black Spot Disease

---

## 📊 Dataset

### 🐟 Fish Species Dataset
Contains images of various commercial and common fish species.
* **Example Species:** Prawn, Black Pomphate, Pomplate etc.
* **Splits:** `Train` | `Validation` | `Test`

### 🦠 Fish Disease Dataset
Contains fish images categorized strictly by disease type.
* **Classes:** Healthy, Diseased
* **Splits:** `Train` | `Validation` | `Test`

---

## 📂 Project Structure

```text
Fish-Species-and-Disease-Detection/
│
├── dataset/
│   ├── fish_species/   (train/ | val/ | test/)
│   └── fish_disease/   (train/ | val/ | test/)
│
├── models/
│   ├── train_species_model.py
│   ├── train_disease_model.py
│   ├── predict_fish.py
│   └── model_weights/
│
├── notebooks/
│   ├── species_training.ipynb
│   └── disease_training.ipynb
│
├── utils/
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation & Setup

**Step 1:** Clone the Repository
```bash
git clone [https://github.com/Tathagato13/Fish-Species-and-Disease-Detection.git](https://github.com/Tathagato13/Fish-Species-and-Disease-Detection.git)
cd Fish-Species-and-Disease-Detection
```

**Step 2:** Create a Virtual Environment
```bash
python -m venv venv
```

**Step 3:** Activate the Environment
* **Windows:**
  ```bash
  venv\Scripts\activate
  ```
* **Linux / Mac:**
  ```bash
  source venv/bin/activate
  ```

**Step 4:** Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

* `tensorflow` | `torch` | `torchvision`
* `opencv-python` | `Pillow`
* `numpy` | `pandas` | `matplotlib`
* `scikit-learn` | `tqdm`

---

## 🏋️ Model Training

**Train the Fish Species Model:**
```bash
python models/train_species_model.py
# Output: models/model_weights/species_model.pth
```

**Train the Fish Disease Model:**
```bash
python models/train_disease_model.py
# Output: models/model_weights/disease_model.pth
```

---

## 🔍 Model Prediction

Run the prediction script on a target image:
```bash
python models/predict_fish.py --image fish.jpg
```

**Example Output:**
```yaml
Input Image: fish.jpg
-----------------------------
Predicted Species : Tilapia
Predicted Disease : White Spot
Confidence        : 92%
```

---

## 📈 Evaluation Metrics

The models are rigorously evaluated using standard classification metrics:
* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

---

## 🚀 Future Improvements

- [ ] Implement **YOLO** for real-time bounding-box fish detection.
- [ ] Develop a **Mobile Application** for on-the-go field testing.
- [ ] Deploy via **Edge AI** on Raspberry Pi / NVIDIA Jetson devices.
- [ ] Direct API integration with **Smart Aquaculture Systems**.

---

## 👨‍💻 Author

**Tathagato** *Electronics and Communication Engineering* *Techno Main Salt Lake* 🔗 **GitHub:** [@Tathagato13](https://github.com/Tathagato13)

---
*If you find this project helpful, please consider giving it a ⭐ on GitHub!*
