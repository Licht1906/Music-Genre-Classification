import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# =========================
# 1. Khai báo mô hình
# =========================
class MusicCNNLSTM(nn.Module):
    """CNN2D + BiLSTM dùng cho demo dự đoán thể loại nhạc."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time_steps, channels * freq_bins)
        _, (hidden, _) = self.lstm(x)
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(x)


# =========================
# 2. Đường dẫn dữ liệu
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
TASK_B_DIR = os.path.join(RESULTS_DIR, "task_b")
TASK_C_DIR = os.path.join(RESULTS_DIR, "task_c")
FINAL_DIR = os.path.join(RESULTS_DIR, "final")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TASK_B_SUMMARY = os.path.join(TASK_B_DIR, "final_summary.csv")
TASK_C_METRICS = os.path.join(TASK_C_DIR, "dl_metrics.csv")
DL_PREDICTIONS = os.path.join(TASK_C_DIR, "dl_predictions.pkl")
LEARNING_CURVE = os.path.join(TASK_C_DIR, "learning_curve.png")

MODEL_COMPARISON = os.path.join(FINAL_DIR, "model_comparison.csv")
CONFUSION_MATRIX = os.path.join(FINAL_DIR, "cnn_lstm_confusion_matrix.png")
ERROR_ANALYSIS = os.path.join(FINAL_DIR, "genre_error_analysis.csv")
CLASSIFICATION_REPORT = os.path.join(FINAL_DIR, "cnn_lstm_classification_report.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_best.pt")

GENRE_NAMES_VI = {
    "blues": "Blues",
    "classical": "Cổ điển",
    "country": "Đồng quê",
    "disco": "Disco",
    "hiphop": "Hip-hop",
    "jazz": "Jazz",
    "metal": "Metal",
    "pop": "Pop",
    "reggae": "Reggae",
    "rock": "Rock",
}


# =========================
# 3. Hàm tiện ích
# =========================
def safe_read_csv(path):
    """Đọc file CSV nếu tồn tại, ngược lại trả về None."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def show_file_warning(path):
    """Hiển thị cảnh báo thiếu file trên dashboard."""
    st.warning(f"Không tìm thấy file: `{path}`")


def format_percent(value):
    """Chuyển số thập phân thành chuỗi phần trăm."""
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


@st.cache_resource
def load_cnn_lstm_model():
    """Load mô hình CNN2D + BiLSTM từ file state_dict."""
    if not os.path.exists(MODEL_PATH):
        return None

    model = MusicCNNLSTM(num_classes=10)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_genres():
    """Load danh sách genre từ file prediction của Task C."""
    if not os.path.exists(DL_PREDICTIONS):
        return None

    with open(DL_PREDICTIONS, "rb") as f:
        pred_data = pickle.load(f)

    return pred_data.get("genres", None)


def plot_mel_spectrogram(mel_db):
    """Vẽ Mel Spectrogram để hiển thị trong tab demo."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel_db, aspect="auto", origin="lower")
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Dải Mel")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    return fig


def predict_audio_genre(uploaded_file):
    """Xử lý file audio upload và trả về genre dự đoán cùng xác suất."""
    import librosa

    genres = load_genres()
    model = load_cnn_lstm_model()

    if genres is None:
        raise FileNotFoundError("Không tìm thấy danh sách genre trong dl_predictions.pkl.")
    if model is None:
        raise FileNotFoundError("Không tìm thấy model cnn_lstm_best.pt.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Đưa Mel Spectrogram về cùng kích thước input mà model yêu cầu.
    target_frames = 130
    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_input = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_input = mel_db[:, :target_frames]

    # Chuẩn hóa đơn giản cho inference.
    mel_input = (mel_input - mel_input.mean()) / (mel_input.std() + 1e-8)

    possible_inputs = [
        torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0),
    ]

    output = None
    last_error = None

    for x in possible_inputs:
        try:
            with torch.no_grad():
                output = model(x)
            break
        except Exception as error:
            last_error = error

    if output is None:
        raise RuntimeError(f"Input shape chưa khớp với model: {last_error}")

    if isinstance(output, tuple):
        output = output[0]

    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

    return {
        "mel_db": mel_db,
        "genres": genres,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_genre": genres[pred_idx],
        "confidence": float(probs[pred_idx]),
    }


# =========================
# 4. Cấu hình giao diện
# =========================
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Dashboard phân loại thể loại nhạc")
st.caption("Task D — Đánh giá mô hình, trực quan hóa kết quả và demo dự đoán")

# Đọc dữ liệu 
task_b = safe_read_csv(TASK_B_SUMMARY)
task_c = safe_read_csv(TASK_C_METRICS)
model_comparison = safe_read_csv(MODEL_COMPARISON)
classification_report_df = safe_read_csv(CLASSIFICATION_REPORT)
error_analysis = safe_read_csv(ERROR_ANALYSIS)

# =========================
# 5. Các tab chính
# =========================
tabs = st.tabs([
    "📌 Tổng quan",
    "🤖 Task B - ML truyền thống",
    "🧠 Task C - Deep Learning",
    "📊 So sánh mô hình",
    "🎧 Demo dự đoán",
])


# =========================
# Tab 1: Tổng quan
# =========================
with tabs[0]:
    st.header("📌 Tổng quan dự án")

    col1, col2, col3 = st.columns(3)
    col1.metric("Bài toán", "Phân loại thể loại nhạc")
    col2.metric("Dataset", "GTZAN")
    col3.metric("Số lớp", "10 genre")

    st.markdown("""
    Dự án xây dựng hệ thống phân loại thể loại nhạc dựa trên dữ liệu âm thanh.
    Nhóm thử nghiệm hai hướng tiếp cận chính:

    - **Task B:** sử dụng các đặc trưng âm thanh dạng bảng và các mô hình Machine Learning truyền thống.
    - **Task C:** sử dụng Mel Spectrogram và mô hình Deep Learning CNN2D + BiLSTM.
    - **Task D:** tổng hợp kết quả, trực quan hóa, so sánh mô hình và xây dựng demo dự đoán.
    """)

    st.subheader("10 thể loại được phân loại")
    genre_cols = st.columns(5)
    genres_display = list(GENRE_NAMES_VI.values())
    for idx, genre in enumerate(genres_display):
        genre_cols[idx % 5].button(genre, use_container_width=True, disabled=True)

    st.subheader("Cấu trúc chính của project")
    st.code("""
Music-Genre-Classification/
├── app/
│   └── dashboard.py
├── data/
├── models/
│   └── cnn_lstm_best.pt
├── results/
│   ├── task_b/
│   ├── task_c/
│   └── final/
└── notebooks/
    └── task-d-evaluation_demo.ipynb
    """)


# =========================
# Tab 2: Task B
# =========================
with tabs[1]:
    st.header("🤖 Task B — Kết quả mô hình Machine Learning truyền thống")

    st.info(
        "Task B sử dụng các đặc trưng âm thanh đã được trích xuất sẵn từ `features_3_sec.csv` "
        "và huấn luyện các mô hình như KNN, SVM, Random Forest, XGBoost."
    )

    if task_b is None:
        show_file_warning(TASK_B_SUMMARY)
    else:
        st.subheader("Bảng kết quả Task B")
        st.dataframe(task_b, use_container_width=True)

        acc_col = None
        for candidate in ["DS1 Test", "DS2 Test", "Accuracy", "accuracy", "Test_Accuracy"]:
            if candidate in task_b.columns:
                acc_col = candidate
                break

        if "Model" in task_b.columns and acc_col:
            chart_df = task_b[["Model", acc_col]].copy()
            chart_df = chart_df.rename(columns={acc_col: "Accuracy"})
            chart_df = chart_df.sort_values("Accuracy", ascending=False)

            best_row = chart_df.iloc[0]
            st.success(
                f"Mô hình ML tốt nhất trong Task B: **{best_row['Model']}** "
                f"với accuracy = **{format_percent(best_row['Accuracy'])}**."
            )

            st.subheader("Biểu đồ accuracy của các mô hình Task B")
            st.bar_chart(chart_df.set_index("Model")["Accuracy"])
        else:
            st.warning("Không tìm thấy đủ cột `Model` và cột accuracy/test accuracy trong file Task B.")


# =========================
# Tab 3: Task C
# =========================
with tabs[2]:
    st.header("🧠 Task C — Kết quả mô hình Deep Learning")

    st.info(
        "Task C sử dụng Mel Spectrogram từ raw audio và mô hình CNN2D + BiLSTM. "
        "CNN học các mẫu cục bộ trên phổ âm, còn BiLSTM học quan hệ theo thời gian."
    )

    if task_c is None:
        show_file_warning(TASK_C_METRICS)
    else:
        st.subheader("Chỉ số đánh giá tổng hợp")
        st.dataframe(task_c, use_container_width=True)

        row = task_c.iloc[0]
        col1, col2, col3 = st.columns(3)

        if "accuracy" in task_c.columns:
            col1.metric("Accuracy", format_percent(row["accuracy"]))
        if "f1_macro" in task_c.columns:
            col2.metric("F1 Macro", format_percent(row["f1_macro"]))
        if "f1_weighted" in task_c.columns:
            col3.metric("F1 Weighted", format_percent(row["f1_weighted"]))

        if "notes" in task_c.columns:
            st.caption(f"Ghi chú mô hình: {row['notes']}")

    st.subheader("Learning Curve")
    if os.path.exists(LEARNING_CURVE):
        st.image(LEARNING_CURVE, use_container_width=True)
    else:
        show_file_warning(LEARNING_CURVE)

    st.subheader("Confusion Matrix — CNN2D + BiLSTM")
    st.caption(
        "Ma trận nhầm lẫn cho biết mô hình phân loại đúng/sai ở từng genre và các genre nào dễ bị nhầm với nhau."
    )
    if os.path.exists(CONFUSION_MATRIX):
        st.image(CONFUSION_MATRIX, use_container_width=True)
    else:
        show_file_warning(CONFUSION_MATRIX)

    st.subheader("Classification Report")
    if classification_report_df is not None:
        st.dataframe(classification_report_df, use_container_width=True)
    else:
        show_file_warning(CLASSIFICATION_REPORT)

    st.subheader("Phân tích lỗi theo từng genre")
    if error_analysis is not None:
        st.dataframe(error_analysis, use_container_width=True)

        if "Genre" in error_analysis.columns and "Class Accuracy" in error_analysis.columns:
            chart_error = error_analysis[["Genre", "Class Accuracy"]].sort_values("Class Accuracy")
            st.bar_chart(chart_error.set_index("Genre")["Class Accuracy"])
    else:
        show_file_warning(ERROR_ANALYSIS)
        st.info("Hãy chạy lại notebook Task D để sinh file `genre_error_analysis.csv`.")


# =========================
# Tab 4: So sánh mô hình
# =========================
with tabs[3]:
    st.header("📊 So sánh tổng hợp các mô hình")

    st.info(
        "Lưu ý: Task B và Task C dùng hai pipeline khác nhau. "
        "Task B dùng tabular features từ `features_3_sec.csv`, còn Task C dùng Mel Spectrogram từ raw audio. "
        "Vì vậy bảng so sánh giúp tham khảo hiệu quả mô hình, nhưng không nên hiểu là hai pipeline hoàn toàn giống nhau."
    )

    if model_comparison is None:
        show_file_warning(MODEL_COMPARISON)
        st.info("Hãy chạy notebook Task D trước để sinh file `model_comparison.csv`.")
    else:
        st.subheader("Bảng so sánh")
        st.dataframe(model_comparison, use_container_width=True)

        if "Model" in model_comparison.columns and "Accuracy" in model_comparison.columns:
            sorted_comparison = model_comparison.sort_values("Accuracy", ascending=False)
            best_row = sorted_comparison.iloc[0]

            col1, col2 = st.columns(2)
            col1.metric("Mô hình tốt nhất", best_row["Model"])
            col2.metric("Accuracy cao nhất", format_percent(best_row["Accuracy"]))

            st.subheader("Biểu đồ so sánh accuracy")
            st.bar_chart(sorted_comparison.set_index("Model")["Accuracy"])

            st.success(
                f"Theo kết quả hiện tại, mô hình tốt nhất là **{best_row['Model']}** "
                f"với accuracy = **{format_percent(best_row['Accuracy'])}**."
            )


# =========================
# Tab 5: Demo dự đoán
# =========================
with tabs[4]:
    st.header("🎧 Demo dự đoán thể loại nhạc")

    st.markdown(
        "Upload một file audio định dạng `.wav`, `.mp3` hoặc `.ogg`. "
        "Hệ thống sẽ trích xuất Mel Spectrogram và dùng mô hình CNN2D + BiLSTM để dự đoán genre."
    )

    uploaded_file = st.file_uploader(
        "Chọn file audio",
        type=["wav", "mp3", "ogg"],
    )

    if uploaded_file is None:
        st.info("Hãy upload một file nhạc để bắt đầu demo.")
    else:
        st.audio(uploaded_file)

        if st.button("🎯 Dự đoán thể loại", type="primary"):
            try:
                result = predict_audio_genre(uploaded_file)

                st.subheader("Mel Spectrogram của file upload")
                st.pyplot(plot_mel_spectrogram(result["mel_db"]))

                genres = result["genres"]
                probs = result["probs"]
                pred_genre = result["pred_genre"]
                confidence = result["confidence"]

                st.success(f"Thể loại dự đoán: **{pred_genre.upper()}**")
                st.metric("Độ tin cậy", format_percent(confidence))

                if confidence < 0.5:
                    st.warning(
                        "Độ tin cậy thấp. File audio có thể khác phân phối dữ liệu GTZAN, "
                        "hoặc mô hình đang phân vân giữa nhiều genre."
                    )

                top3_idx = np.argsort(probs)[::-1][:3]
                top3_df = pd.DataFrame({
                    "Hạng": [1, 2, 3],
                    "Genre": [genres[i] for i in top3_idx],
                    "Xác suất": [float(probs[i]) for i in top3_idx],
                    "Xác suất (%)": [float(probs[i]) * 100 for i in top3_idx],
                })

                st.subheader("Top 3 dự đoán")
                st.dataframe(top3_df, use_container_width=True)

                prob_df = pd.DataFrame({
                    "Genre": genres,
                    "Xác suất": probs,
                }).sort_values("Xác suất", ascending=False)
                prob_df["Xác suất"] = prob_df["Xác suất"].astype(float)
                prob_df["Xác suất (%)"] = prob_df["Xác suất"] * 100

                st.subheader("Xác suất cho từng genre")
                st.dataframe(prob_df, use_container_width=True)
                st.bar_chart(prob_df.set_index("Genre")["Xác suất"])

            except Exception as error:
                st.error("Demo dự đoán bị lỗi. Các phần dashboard khác vẫn sử dụng được.")
                st.exception(error)
