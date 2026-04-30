import os

# ── Đường dẫn ───────────────────────────────────────────────
BASE_DIR        = "/kaggle/working/gtzan_project" 
DATA_RAW_DIR    = "/kaggle/input/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/Data/genres_original"
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR      = os.path.join(BASE_DIR, "data", "splits")
FEATURES_DIR    = os.path.join(BASE_DIR, "features")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ── Dữ liệu ─────────────────────────────────────────────────
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
N_CLASSES       = 10
SAMPLE_RATE     = 22050
DURATION        = 30        # giây
RANDOM_SEED     = 42

# ── Đặc trưng âm thanh ──────────────────────────────────────
N_MFCC          = 40
N_MELS          = 128
HOP_LENGTH      = 512
N_FFT           = 2048
MEL_IMG_SIZE    = (128, 128)  # resize cho CNN

# ── Train/Val/Test split ─────────────────────────────────────
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

# ── Training ─────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 0.001