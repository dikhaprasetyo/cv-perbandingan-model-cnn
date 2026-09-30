import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import transforms, models

# ============================================================
# MAPPING NAMA KELAS UNTUK TAMPILAN FRONTEND
# ============================================================

LABEL_MAPPING = {
    "glass": "Botol Kaca",
    "kaleng": "Botol Kaleng",
    "metal": "Botol Kaleng",
    "can": "Botol Kaleng",
    "cans": "Botol Kaleng",
    "plastik": "Botol Plastik",
    "plastic": "Botol Plastik"
}


def format_label(label):
    label_lower = label.lower()
    return LABEL_MAPPING.get(label_lower, label.replace("_", " ").title())


# ============================================================
# KONFIGURASI
# ============================================================

SAVED_MODEL_DIR = "saved_models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# STYLE FRONTEND
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Citra CNN",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label, span {
        color: white !important;
    }

    .main-box {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
    }

    .result-box {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-top: 15px;
    }

    .metric-box {
        background-color: #111827;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background-color: #161b22;
        border-radius: 12px;
        padding: 10px;
        border: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEMBUAT ARSITEKTUR MODEL
# ============================================================

def build_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            num_classes
        )

    elif model_name == "inceptionv3":
        model = models.inception_v3(weights=None, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(
                model.AuxLogits.fc.in_features,
                num_classes
            )

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    elif model_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )

    else:
        raise ValueError(f"Model tidak dikenali: {model_name}")

    return model


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    input_size = checkpoint["input_size"]

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, model_name, class_names, input_size


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image, input_size):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


# ============================================================
# PREDIKSI
# ============================================================

def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = format_label(class_names[predicted_idx.item()])
        confidence_score = confidence.item() * 100

    return predicted_class, confidence_score, probabilities.cpu().numpy()[0]


# ============================================================
# CEK MODEL TERSEDIA
# ============================================================

def get_available_models():
    if not os.path.exists(SAVED_MODEL_DIR):
        return []

    model_files = [
        file for file in os.listdir(SAVED_MODEL_DIR)
        if file.endswith(".pth")
    ]

    return model_files


# ============================================================
# TAMPILAN UTAMA
# ============================================================

st.title("Klasifikasi Citra Menggunakan Model CNN")
st.write(
    "Frontend ini digunakan untuk melakukan prediksi gambar menggunakan model "
    "ResNet18, ResNet50, VGG16, InceptionV3, DenseNet121, dan MobileNetV3 "
    "yang telah dilatih sebelumnya."
)

st.markdown("---")

available_models = get_available_models()

if len(available_models) == 0:
    st.error(
        "Folder saved_models belum ditemukan atau belum berisi file model .pth. "
        "Jalankan training terlebih dahulu sampai menghasilkan file model."
    )
    st.stop()


col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("1. Pilih Model")

    selected_model_file = st.selectbox(
        "Pilih model yang ingin digunakan:",
        available_models,
        index=available_models.index("resnet50.pth") if "resnet50.pth" in available_models else 0
    )

    model_path = os.path.join(SAVED_MODEL_DIR, selected_model_file)

    model, model_name, class_names, input_size = load_model(model_path)

    st.info(f"Model aktif: {model_name.upper()}")
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Ukuran input: `{input_size} x {input_size}`")
    st.write(f"Jumlah kelas: `{len(class_names)}`")
    display_class_names = [format_label(label) for label in class_names]
    st.write(f"Kelas: {', '.join(display_class_names)}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("2. Upload Gambar")

    uploaded_file = st.file_uploader(
        "Pilih gambar untuk diprediksi",
        type=["jpg", "jpeg", "png", "webp"]
    )

    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)

    st.subheader("3. Hasil Prediksi")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        image_tensor = preprocess_image(image, input_size)

        predicted_class, confidence_score, probabilities = predict_image(
            model,
            image_tensor,
            class_names
        )

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.success("Prediksi berhasil dilakukan.")

        st.markdown(
            f"""
            <div class="metric-box">
                <h2>{predicted_class}</h2>
                <p>Confidence: {confidence_score:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Probabilitas Setiap Kelas")

        isplay_class_names = [format_label(label) for label in class_names]

        probability_data = {
        "Kelas": display_class_names,
        "Probabilitas (%)": [round(float(prob) * 100, 2) for prob in probabilities]
        }

        st.dataframe(
            probability_data,
            use_container_width=True
        )

        st.bar_chart(
        data={
        display_class_names[i]: float(probabilities[i]) * 100
        for i in range(len(display_class_names))
    }
)

    else:
        st.warning("Silakan upload gambar terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

st.caption(
    "Aplikasi frontend ini menggunakan Streamlit dan model CNN hasil training "
    "dari folder saved_models."
)