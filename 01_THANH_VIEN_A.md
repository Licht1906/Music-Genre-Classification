# 👤 Thành Viên A — Data Engineer

> **Nhiệm vụ:** Thu thập & hiểu dữ liệu, tiền xử lý audio, data augmentation, tạo train/val/test split chuẩn cho cả nhóm dùng chung.

---

## 🎯 Mục Tiêu Cần Đạt

- [ ] Load và kiểm tra toàn bộ GTZAN dataset (1000 files)
- [ ] Thực hiện EDA — phân tích phân phối, trực quan hóa waveform & spectrogram
- [ ] Áp dụng ít nhất 3 kỹ thuật data augmentation trên audio
- [ ] Tạo file `train.csv`, `val.csv`, `test.csv` chuẩn hóa cho cả nhóm
- [ ] Viết `src/data_utils.py` để nhóm tái sử dụng

---

## 📋 Phần 1 — Setup & Load Dữ Liệu

```python
# notebook: A_data_preprocessing.ipynb
# Cell 1: Import
import os, librosa, librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/kaggle/working/gtzan_project')
from src.config import *

# Tạo thư mục nếu chưa có
for d in [DATA_PROC_DIR, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("✅ Setup hoàn tất")
```

```python
# Cell 2: Quét toàn bộ dataset → tạo DataFrame index
records = []
for genre in GENRES:
    folder = os.path.join(DATA_RAW_DIR, genre)
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.wav'):
            records.append({
                "path": os.path.join(folder, fname),
                "genre": genre,
                "label": GENRES.index(genre),
                "filename": fname
            })

df = pd.DataFrame(records)
print(f"Tổng số file: {len(df)}")
print(df['genre'].value_counts())
```

---

## 📊 Phần 2 — Exploratory Data Analysis (EDA)

```python
# Cell 3: Kiểm tra thời lượng thực tế của mỗi file
durations = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking duration"):
    try:
        y, sr = librosa.load(row['path'], sr=SAMPLE_RATE, duration=DURATION)
        durations.append(len(y) / sr)
    except Exception as e:
        durations.append(None)
        print(f"⚠️ Lỗi file: {row['path']} — {e}")

df['duration'] = durations
print(df['duration'].describe())
# Kiểm tra file ngắn hơn 29s (có thể bị lỗi)
print(f"\nFile có thời lượng < 29s: {(df['duration'] < 29).sum()}")
```

```python
# Cell 4: Visualize waveform & mel spectrogram cho mỗi thể loại
fig, axes = plt.subplots(10, 2, figsize=(14, 30))
fig.suptitle("Waveform & Mel Spectrogram theo từng thể loại", fontsize=14, y=1.01)

for i, genre in enumerate(GENRES):
    sample = df[df['genre'] == genre].iloc[0]
    y, sr = librosa.load(sample['path'], sr=SAMPLE_RATE, duration=10)

    # Waveform
    axes[i, 0].set_title(f"{genre} — Waveform", fontsize=10)
    librosa.display.waveshow(y, sr=sr, ax=axes[i, 0])
    axes[i, 0].set_xlabel("")

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH,
                                   x_axis='time', y_axis='mel', ax=axes[i, 1])
    axes[i, 1].set_title(f"{genre} — Mel Spectrogram", fontsize=10)
    fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_waveform_spectrogram.png"), dpi=100, bbox_inches='tight')
plt.show()
```

```python
# Cell 5: Phân phối nhãn (kiểm tra class imbalance)
plt.figure(figsize=(10, 4))
df['genre'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
plt.title("Phân phối số lượng file theo thể loại")
plt.xlabel("Thể loại"); plt.ylabel("Số lượng")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_class_distribution.png"), dpi=100)
plt.show()
# GTZAN cân bằng (100 file/thể loại) → không cần oversampling
```

---

## 🔊 Phần 3 — Data Augmentation

> **Lý do quan trọng:** Augmentation giúp mô hình tổng quát tốt hơn, tránh overfit. Đây là điểm khác biệt so với làm đơn giản.

```python
# Cell 6: Định nghĩa các hàm augmentation
import random

def augment_time_stretch(y, rate_range=(0.8, 1.2)):
    """Kéo dãn hoặc nén thời gian — không thay đổi pitch"""
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate=rate)

def augment_pitch_shift(y, sr, steps_range=(-3, 3)):
    """Thay đổi cao độ (pitch) ±3 semitones"""
    steps = random.randint(*steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def augment_add_noise(y, noise_factor=0.005):
    """Thêm Gaussian noise nhỏ vào tín hiệu"""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def apply_random_augmentation(y, sr):
    """Áp dụng ngẫu nhiên 1 trong 3 kỹ thuật"""
    choice = random.choice(['time_stretch', 'pitch_shift', 'noise'])
    if choice == 'time_stretch':
        y_aug = augment_time_stretch(y)
    elif choice == 'pitch_shift':
        y_aug = augment_pitch_shift(y, sr)
    else:
        y_aug = augment_add_noise(y)
    # Đảm bảo độ dài bằng gốc
    target_len = int(DURATION * sr)
    if len(y_aug) > target_len:
        y_aug = y_aug[:target_len]
    else:
        y_aug = np.pad(y_aug, (0, target_len - len(y_aug)))
    return y_aug, choice
```

```python
# Cell 7: Demo augmentation trực quan
sample_path = df[df['genre'] == 'jazz'].iloc[0]['path']
y_orig, sr = librosa.load(sample_path, sr=SAMPLE_RATE, duration=DURATION)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
titles = ['Original', 'Time Stretch (rate=0.85)', 'Pitch Shift (+2 semitones)', 'Add Noise']
signals = [
    y_orig,
    augment_time_stretch(y_orig, rate_range=(0.85, 0.85)),
    augment_pitch_shift(y_orig, sr, steps_range=(2, 2)),
    augment_add_noise(y_orig)
]

for ax, sig, title in zip(axes, signals, titles):
    librosa.display.waveshow(sig[:sr*5], sr=sr, ax=ax)  # hiện 5 giây đầu
    ax.set_title(title); ax.set_xlabel("")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "eda_augmentation_demo.png"), dpi=100)
plt.show()
```

---

## ✂️ Phần 4 — Tạo Train/Val/Test Split

```python
# Cell 8: Stratified split — đảm bảo tỉ lệ nhãn đều nhau
from sklearn.model_selection import train_test_split

# Bước 1: Tách test ra trước
df_trainval, df_test = train_test_split(
    df, test_size=TEST_RATIO, stratify=df['label'], random_state=RANDOM_SEED
)

# Bước 2: Tách train và val từ phần còn lại
val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
df_train, df_val = train_test_split(
    df_trainval, test_size=val_ratio_adjusted,
    stratify=df_trainval['label'], random_state=RANDOM_SEED
)

print(f"Train: {len(df_train)} files ({len(df_train)/len(df)*100:.1f}%)")
print(f"Val:   {len(df_val)} files ({len(df_val)/len(df)*100:.1f}%)")
print(f"Test:  {len(df_test)} files ({len(df_test)/len(df)*100:.1f}%)")

# Kiểm tra phân phối nhãn đồng đều
print("\nPhân phối nhãn trong Train:")
print(df_train['genre'].value_counts().sort_index())
```

```python
# Cell 9: Lưu splits
df_train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
df_val.to_csv(os.path.join(SPLITS_DIR, "val.csv"),   index=False)
df_test.to_csv(os.path.join(SPLITS_DIR, "test.csv"),  index=False)

print("✅ Đã lưu train.csv, val.csv, test.csv vào", SPLITS_DIR)
print("📌 Thông báo nhóm: B và C có thể bắt đầu dùng các file split này!")
```

---

## 🛠️ Phần 5 — Viết `src/data_utils.py`

> File này cả nhóm sẽ dùng để load dữ liệu nhất quán.

```python
# src/data_utils.py
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import *


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=0.0):
    """
    Load file audio. Trả về (y, sr).
    Tự động padding nếu file ngắn hơn duration.
    """
    try:
        y, sr_orig = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        target_len = int(duration * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return y, sr
    except Exception as e:
        print(f"⚠️ Lỗi load file {file_path}: {e}")
        return np.zeros(int(duration * sr)), sr


def load_split(split_name):
    """
    Load một split ('train', 'val', 'test').
    Trả về DataFrame với cột: path, genre, label, filename.
    """
    path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa tìm thấy {path}. Hãy chạy notebook của A trước!")
    return pd.read_csv(path)


def genre_to_label(genre):
    return GENRES.index(genre)


def label_to_genre(label):
    return GENRES[label]


def get_all_audio_paths():
    """Trả về danh sách tất cả file audio trong dataset gốc."""
    paths = []
    for genre in GENRES:
        folder = os.path.join(DATA_RAW_DIR, genre)
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.wav'):
                paths.append((os.path.join(folder, fname), genre))
    return paths
```

---

## ✅ Checklist Trước Khi Bàn Giao Cho Nhóm

- [ ] `train.csv`, `val.csv`, `test.csv` đã có trong `data/splits/`
- [ ] Mỗi file có đầy đủ cột: `path`, `genre`, `label`, `filename`
- [ ] Phân phối nhãn đồng đều trong cả 3 splits
- [ ] `src/data_utils.py` đã hoàn thiện và test chạy được
- [ ] Đã lưu EDA plots vào `results/`
- [ ] Không có file audio bị lỗi (kiểm tra bằng Cell 3)
- [ ] Thông báo nhóm: B và C có thể bắt đầu

---

## 💡 Ghi Chú Kỹ Thuật

**Tại sao dùng `sr=22050`?**
Đây là sample rate chuẩn trong music analysis. Đủ để bắt hài âm quan trọng (lên đến 11 kHz theo Nyquist) mà file size không quá lớn.

**Tại sao stratified split?**
Đảm bảo mỗi thể loại đều xuất hiện đều trong train/val/test. Nếu random thông thường, có thể xảy ra trường hợp val/test thiếu một thể loại nào đó.

**Khi nào augment?**
Chỉ augment trên **train set**. Val và Test phải giữ nguyên bản gốc để đánh giá khách quan. B sẽ thực hiện augment trong quá trình trích xuất đặc trưng.
