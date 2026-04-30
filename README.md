# 🎵 GTZAN Music Genre Classification — Tổng Quan Nhóm

> **Mục tiêu:** Xây dựng hệ thống phân loại giai điệu âm nhạc từ bộ dữ liệu GTZAN sử dụng Machine Learning, so sánh hiệu quả của 5 mô hình khác nhau và deploy demo app.

---

## 👥 Phân Công Nhóm

| Thành viên | Vai trò chính | Phụ trách |
|---|---|---|
| Hùng | Data Engineer | Thu thập dữ liệu, tiền xử lý, augmentation, EDA |
| **B** | ML Engineer | Trích xuất đặc trưng, train SVM & Random Forest |
| **C** | Deep Learning Engineer | Xây dựng CNN, LSTM, CNN+LSTM hybrid |
| **D** | Evaluation & Deployment | Đánh giá mô hình, visualization, Gradio app, báo cáo |

---
Lưu ý: Không cần thiết tải dữ liệu về, dataset tận 1.4 GB. Lên Kaggle làm, anh em làm cái nào thì import notebook của mình lên kaggle, sau đó tải data cần thiết ở thư mục data và cần thiết add dataset gốc thì add bằng url: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification vào notebook kaggle
## 📁 Cấu Trúc Thư Mục Chung

```
gtzan_project/
│
├── data/
│   ├── raw/                    # Dữ liệu gốc từ GTZAN: 10 thư mục: blues, ...
│   ├── processed/              # Audio sau augmentation
│   └── splits/                 # train.csv, val.csv, test.csv
│
├── features/
│   ├── mfcc_features.npy       # MFCC đã trích xuất
│   ├── mel_spectrograms/       # Ảnh mel spectrogram
│   └── feature_matrix.csv      # Tổng hợp tất cả đặc trưng
│
├── models/
│   ├── svm_model.pkl           # Model SVM đã train
│   ├── rf_model.pkl            # Model Random Forest
│   ├── cnn_model.h5            # Model CNN
│   ├── lstm_model.h5           # Model LSTM
│   └── cnn_lstm_model.h5       # Model hybrid
│
├── notebooks/
│   ├── A_data_preprocessing.ipynb
│   ├── B_feature_ml_models.ipynb
│   ├── C_deep_learning.ipynb
│   └── D_evaluation_demo.ipynb
│
├── src/
│   ├── config.py               # Cấu hình chung
│   ├── data_utils.py           # Hàm dùng chung (A viết)
│   ├── feature_utils.py        # Hàm đặc trưng (B viết)
│   └── model_utils.py          # Hàm evaluate (D viết)
│
├── results/
│   ├── confusion_matrices/
│   ├── tsne_plots/
│   └── metrics_summary.csv
│
├── app/
│   └── gradio_app.py           # Demo app (D viết)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ File Cấu Hình Chung (`src/config.py`)
> **Tất cả thành viên phải dùng chung file này** để đảm bảo nhất quán.
# src/config.py

---

## 📦 Cài Đặt Môi Trường
# requirements.txt 
pip install -r requirements.txt

---

## 📅 Timeline & Mốc Quan Trọng

```
Tuần 1-2  │ A: EDA + Augmentation ─────────────────────────────┐
           │ B: Setup feature pipeline ──────────────────────┐  │
           │                                                  │  │
Tuần 3-4  │ A: Hoàn thiện splits ───────────────────────────┘  │
           │ B: Train SVM + RF ──────────────────────────────┐  │
           │ C: Thiết kế kiến trúc CNN ──────────────────────┐  │
           │                                                  │  │
Tuần 5-6  │ B: Tối ưu SVM/RF, ghi kết quả ─────────────────┘  │
           │ C: Train LSTM, CNN+LSTM ────────────────────────┘  │
           │ D: Setup evaluation framework ──────────────────┐  │
           │                                                  │  │
Tuần 7    │ C: Hoàn thiện deep learning ──────────────────── │ ─┘
           │ D: Chạy đánh giá toàn bộ mô hình ───────────────┤
           │                                                  │
Tuần 8    │ D: Gradio app + báo cáo tổng kết ────────────────┘
           │ Tất cả: Review, merge, present
```

---

## 🤝 Quy Tắc Làm Việc Nhóm

**1. Giao tiếp kết quả:**
- Mỗi thành viên lưu model/feature ra đúng thư mục theo `config.py`
- Ghi kết quả metrics vào `results/metrics_summary.csv` theo format: `model, accuracy, f1_macro, f1_weighted`

**2. Kiểm tra tính nhất quán:**
- Luôn dùng `RANDOM_SEED = 42` khi split data, khởi tạo model
- B và C phải dùng **cùng một bộ train/val/test** do Hùng tạo ra

**3. Notebook convention:**
- Cell đầu tiên: `from src.config import *`
- Có markdown giải thích từng bước
- Chạy sạch từ đầu đến cuối (Restart & Run All) trước khi nộp

**4. Khi gặp lỗi:**
- Lỗi data → hỏi Hùng
- Lỗi feature → hỏi B  
- Lỗi model deep learning → hỏi C
- Lỗi metrics/visualization → hỏi D

---

## 📊 Format Lưu Kết Quả Chung

Mỗi thành viên sau khi train xong, **bắt buộc** ghi vào file này:

```python
# Ví dụ cách ghi kết quả vào metrics_summary.csv
import pandas as pd, os

def save_metrics(model_name, accuracy, f1_macro, f1_weighted, notes=""):
    row = {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "notes": notes
    }
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)
    print(f"✅ Đã lưu kết quả: {model_name} — Accuracy: {accuracy:.4f}")
```
