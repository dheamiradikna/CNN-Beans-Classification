# 🌱 CNN Image Classification — Beans Disease Dataset

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Dataset](https://img.shields.io/badge/Dataset-Beans%20Disease-green)
![Accuracy](https://img.shields.io/badge/Accuracy-%E2%89%A585%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Proyek klasifikasi penyakit daun kacang menggunakan **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras. Model mampu membedakan 3 kondisi daun kacang berdasarkan citra.

---

## 📂 Struktur Direktori

```
CNN-Beans-Classification/
├── tfjs_model/
│   ├── group1-shard1of1.bin   ← bobot model (binary)
│   └── model.json             ← arsitektur model untuk browser
├── tflite/
│   ├── model.tflite           ← model terkompresi untuk mobile/edge
│   └── label.txt              ← daftar label kelas
├── saved_model/
│   ├── saved_model.pb         ← graph model TensorFlow
│   └── variables/             ← bobot model
├── notebook.ipynb             ← notebook training lengkap
├── README.md                  ← dokumentasi ini
└── requirements.txt           ← dependensi Python
```

---

## 🗂️ Dataset

| Atribut | Detail |
|---------|--------|
| Nama | Beans Disease Dataset |
| Sumber | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) (Hugging Face) |
| Total Gambar | 1.295 gambar |
| Jumlah Kelas | 3 kelas |
| Resolusi | Tidak seragam (bervariasi) |
| Split | Train 70% / Validation 15% / Test 15% |

### 🏷️ Kelas

| Label | Deskripsi | Contoh |
|-------|-----------|--------|
| `angular_leaf_spot` | Penyakit bercak sudut daun | Bercak coklat bersudut pada daun |
| `bean_rust` | Karat kacang | Bercak oranye/coklat seperti karat |
| `healthy` | Daun sehat | Daun hijau tanpa bercak |

---

## 🏗️ Arsitektur Model

```
Input (224×224×3)
  ↓
Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
  ↓
Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
  ↓
Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.25)
  ↓
Conv2D(256) → BN → MaxPool → Dropout(0.25)
  ↓
Flatten → Dense(512) → BN → Dropout(0.5)
  ↓
Dense(256) → Dropout(0.3)
  ↓
Dense(3, softmax)
```

---

## ✅ Checklist Kriteria

| # | Kriteria | Status |
|---|----------|--------|
| 1 | Dataset bebas, min. 1.000 gambar | ✅ 1.295 gambar |
| 2 | Bukan dataset RPS / X-Ray | ✅ Beans Disease |
| 3 | Split Train / Validation / Test | ✅ 70 / 15 / 15 |
| 4 | Sequential + Conv2D + Pooling Layer | ✅ 4 blok Conv2D + MaxPooling2D |
| 5 | Akurasi train & test ≥ 85% | ✅ |
| 6 | Plot akurasi & loss | ✅ |
| 7 | SavedModel + TF-Lite + TFJS | ✅ |
| + | Callback (EarlyStopping, ReduceLR, Checkpoint) | ✅ |
| + | Resolusi gambar tidak seragam | ✅ |
| + | Min. 3 kelas | ✅ 3 kelas |
| + | Inference TF-Lite + bukti output | ✅ |

---

## 🚀 Cara Menjalankan

### 1. Clone repo
```bash
git clone https://github.com/dheamiradikna/CNN-Beans-Classification.git
cd CNN-Beans-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan notebook
Buka di **Google Colab** (disarankan — butuh GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dheamiradikna/CNN-Beans-Classification/blob/main/notebook.ipynb)

---

## 🔍 Cara Inference dengan TF-Lite

```python
import tensorflow as tf
import numpy as np
from PIL import Image

LABELS = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Load model
interpreter = tf.lite.Interpreter(model_path='tflite/model.tflite')
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

# Preprocessing
img = Image.open('foto_daun.jpg').convert('RGB').resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# Inference
interpreter.set_tensor(inp[0]['index'], img_input)
interpreter.invoke()
output = interpreter.get_tensor(out[0]['index'])[0]

print(f'Prediksi   : {LABELS[np.argmax(output)]}')
print(f'Confidence : {np.max(output)*100:.2f}%')
```

---

## 🔧 Callbacks yang Digunakan

| Callback | Parameter | Fungsi |
|----------|-----------|--------|
| `EarlyStopping` | `patience=15`, monitor `val_accuracy` | Hentikan training jika tidak ada peningkatan |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=5` | Kurangi learning rate saat stagnan |
| `ModelCheckpoint` | `save_best_only=True` | Simpan bobot terbaik otomatis |

---

## 📦 Format Model

| Format | File | Kegunaan |
|--------|------|----------|
| SavedModel | `saved_model/` | TF Serving / training lanjutan |
| TF-Lite | `tflite/model.tflite` | Android, iOS, Raspberry Pi |
| TensorFlow.js | `tfjs_model/model.json` | Web browser / Node.js |

---

## 📋 Requirements

```
packaging>=24.2.0
tensorflow>=2.12.0
datasets>=2.14.0
tensorflowjs>=4.10.0
numpy>=1.23.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
seaborn>=0.12.0
Pillow>=9.5.0
```

---

## 👩‍💻 Author

**Dhea Miradikna** — Universitas Negeri Semarang (UNNES)

---

## 📄 License

MIT License — bebas digunakan untuk keperluan edukasi.
