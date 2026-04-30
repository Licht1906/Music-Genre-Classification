import os
import librosa
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from src.config import *


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=0.0):
    """
    Load file audio. Trả về (mảng các mẫu âm thanh dạng np.array có kích thước sr x duration, sr - Sample rate).
    Tự động padding nếu file ngắn hơn duration.
    Nếu có lỗi khi load file, trả về mảng zeros.
    sr - sample rate: là số mẫu âm thanh được lấy mỗi giây. Mỗi mẫu có một giá trị số
    duration: thời lượng đoạn âm thanh muốn lấy tính từ đầu (tính theo giây)
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
    
#Các hàm augumentation
#Tham số y nhận vào ở các hàm dưới đây là mảng numpy chứa dữ liệu âm thanh trả về từ hàm load_audio.
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

# Các hàm tiện ích khác

def load_split(split_name):
    """
    Load train set / validation set/ test set ('train', 'val', 'test').
    Trả về DataFrame với cột: path, genre, label, filename.
    """
    path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa tìm thấy {path}. Hãy chạy notebook data_preprocessing.ipynb trước!")
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