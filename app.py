import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from mapping_info import soil_info

# Pastikan TensorFlow menggunakan versi yang benar
try:
    assert tf.__version__.startswith('2')
except AssertionError:
    st.error("Harap menggunakan TensorFlow versi 2.x atau lebih baru.")

# Jika TensorFlow mengeluarkan peringatan, nonaktifkan log info atau peringatan
tf.get_logger().setLevel('ERROR')  # Hanya tampilkan error, hindari peringatan

# Load model
model = tf.keras.models.load_model("model_terrascan.keras")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# UI
st.title("🌍 TerraScan – Klasifikasi Jenis Tanah")
st.write("Upload foto tanah, dan sistem akan memprediksi jenis tanah, status kesuburannya, serta memberikan penjelasan.")

uploaded_file = st.file_uploader("Upload Gambar Tanah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca dan tampilkan gambar yang diunggah
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocessing Gambar
    img_resized = image.resize((224, 224))  # Ubah ukuran gambar ke 224x224 (untuk model EfficientNet)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)  # Mengonversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Preprocessing sesuai model EfficientNet

    # Prediksi
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])  # Mendapatkan index prediksi dengan probabilitas tertinggi
    confidence = predictions[0][pred_index]  # Confidence score dari prediksi
    predicted_label = labels[pred_index]  # Label yang diprediksi

    # Ambil informasi terkait dari soil_info
    info = soil_info.get(predicted_label, {})
    status = "🌱 Subur" if info.get("subur", False) else "🚫 Tidak Subur"
    description = info.get("deskripsi", "-")

    # Tampilkan hasil
    st.markdown(f"### Jenis Tanah: **{predicted_label}**")
    st.markdown(f"### Status: **{status}**")
    st.markdown(f"**Deskripsi:** {description}")
    st.markdown(f"📊 Confidence: `{confidence:.2%}`")
