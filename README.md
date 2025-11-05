# 🏭 Metal Surface Defect Classification using CNNs

A lightweight convolutional neural network (CNN) for detecting and classifying **metal surface defects**, optimized for research, education, and potential embedded deployment.

---

## 📘 Overview

This project uses convolutional neural networks (CNNs) to classify **6 types of metal surface defects** from the [NEU Metal Surface Defects Dataset](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data).

It supports:
- 🔹 Custom lightweight CNN architectures (`SmallCNN`)
- 🔹 Full ResNet-based models for comparison
- 🔹 Training, evaluation, and inference pipelines
- 🔹 TensorBoard logging
- 🔹 Automatic checkpointing and model comparison logging (`model_results.csv`)

---

## 🧰 Dataset

**Dataset**: [NEU Metal Surface Defects Dataset (Kaggle)](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)

**Description**:
> The NEU dataset contains grayscale images of typical surface defects on hot-rolled steel strips, classified into six categories:
> 1. Crazing  
> 2. Inclusion  
> 3. Patches  
> 4. Pitted surface  
> 5. Rolled-in scale  
> 6. Scratches

Each class contains **300 images** of size **200×200 pixels**.

**Recommended structure after downloading:**
```

data/
├── train/
│   ├── Crazing/
│   ├── Inclusion/
│   ├── Patches/
│   ├── Pitted/
│   ├── Rolled/
│   └── Scratches/
├── valid/
│   └── ...
└── test/
└── ...

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/mohsensalehi1984/MetalSurfaceDefectsNN.git
cd MetalSurfaceDefectsNN
````

### 2️⃣ Create and activate a virtual environment

It is strongly recommended to use a **Python virtual environment** for dependency isolation.

```bash
python3 -m venv venv
source venv/bin/activate
```

(Use `venv\Scripts\activate` on Windows)

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

Train the CNN model from scratch:

```bash
python src/train.py
```

Training progress and loss/accuracy metrics are logged in **TensorBoard**:

```bash
tensorboard --logdir runs/
```

---

## 🧪 Evaluate / Inference

### Evaluate full test set:

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt --evaluate
```

### Predict a single image:

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt --image data/test/Crazing/img_01.png
```

---

## 📊 Model Comparison

Each trained model’s results are logged in `model_results.csv`:

| Timestamp           | Model    | Parameters | Val_Acc |
| ------------------- | -------- | ---------- | ------- |
| 2025-10-05 22:45:03 | SmallCNN | 9,894      | 0.7361  |

You can quickly compare architectures based on parameter count and validation accuracy.

---

## 💡 Notes for Embedded / Edge Deployment

When deploying to embedded or IoT devices:

* Optimize for **parameter count** and **FLOPs**
* Use **8-bit quantization** (e.g., via PyTorch quantization or ONNX)
* Avoid large layers like `Conv2d(512, ...)` or ResNets unless you have an accelerator
* Smaller CNNs (like `SmallCNN`) balance performance and memory footprint well

---

## 📁 Project Structure

```
MetalSurfaceDefectsNN/
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── model.py
│   ├── dataset.py
│   ├── config.py
│   ├── utils.py
│   └── modelInspect.py
├── checkpoints/
│   └── best_model.pt
├── data/
│   └── (train, valid, test)
├── run.sh
├── requirements.txt
└── README.md
```

---

## 📈 Example Results

| Model                    | Parameters | Val Accuracy |
| ------------------------ | ---------- | ------------ |
| SmallCNN (2 conv layers) | 9,894      | 0.7361       |
| SmallCNN (3 conv layers) | 32,614     | 0.8500       |
| ResNet18 (fine-tuned)    | 11.2M      | 0.9800       |

---


## 👨‍💻 Author & License

**Author:** *Mohsen Salehi*

**License:** MIT License

---

