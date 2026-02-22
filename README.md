# 🧠 Deep Learning Image Classification Project

A modular **Deep Learning Image Classification pipeline** built using **TensorFlow & Python**, designed with a clean architecture, configurable training, and scalable experimentation workflow.

This project demonstrates an end-to-end deep learning system including:

✅ Data Loading
✅ Image Preprocessing & Augmentation
✅ Model Building
✅ Training Pipeline
✅ Evaluation & Prediction

---

## 🚀 Project Overview

This repository implements a complete deep learning workflow for image classification using a structured and reusable codebase.

The pipeline automatically:

* Loads dataset from configured path
* Applies preprocessing and augmentation
* Builds a neural network model
* Trains using configurable hyperparameters
* Evaluates model performance

The modular design makes experimentation easy and production-style development possible.

---

## 🏗️ Project Structure

```
Image_classification/
│
├── config/
│   └── config.yaml        # Training configuration
│
├── src/
│   ├── data_loader.py     # Dataset loading
│   ├── preprocessing.py   # Image augmentation
│   ├── model.py           # Model architecture
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Model evaluation
│   ├── predict.py         # Inference script
│   └── utils.py           # Helper utilities
│
├── main.py                # Main training entry point
├── requirements.txt       # Project dependencies
└── README.md
```

---

## ⚙️ Technologies Used

* 🧠 TensorFlow / Keras
* 🐍 Python
* 📊 Scikit-learn
* 🖼 OpenCV
* 📈 Matplotlib
* ⚡ tqdm
* YAML Configuration

---

## 🔧 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/rashuuu2/Image_classification.git
cd Image_classification
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Model

Run the main training pipeline:

```bash
python main.py
```

The script will:

* Load configuration from `config/config.yaml`
* Prepare training & validation datasets
* Build and compile the deep learning model
* Train the network
* Evaluate model performance

---

## 🧩 Configuration

All hyperparameters are controlled using:

```
config/config.yaml
```

Example configurable parameters:

* Dataset path
* Image size
* Batch size
* Learning rate
* Number of epochs

This allows experimentation without modifying source code.

---

## 📊 Model Workflow

```
Dataset
   ↓
Preprocessing & Augmentation
   ↓
Model Architecture
   ↓
Training
   ↓
Evaluation
```

This structure follows real-world machine learning engineering practices.

---

## 📈 Future Improvements

* [ ] TensorBoard visualization
* [ ] Model checkpoint tracking
* [ ] Docker containerization
* [ ] FastAPI deployment
* [ ] HuggingFace model hosting

---

## 🤝 Contributing

Contributions and suggestions are welcome!

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

## 👩‍💻 Author

**Rashi**  
Deep Learning & Machine Learning Enthusiast 🚀


---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
