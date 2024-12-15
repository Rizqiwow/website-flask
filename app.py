import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import io
import matplotlib.pyplot as plt

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model yang sudah disimpan
model = load_model('C:/Users/UsEr/Downloads/tumor_flask/model_namee.keras')  # Sesuaikan dengan path file model Anda

# Fungsi untuk memuat dan memproses gambar
def load_uploaded_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # Resize gambar
    img_array = np.array(img)
    
    # Jika grayscale, ubah ke RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    img_array = img_array / 255.0  # Normalisasi piksel
    return img, img_array  # Return gambar asli dan array

# Fungsi prediksi
def predict_image(image_bytes):
    # Muat gambar
    img, img_array = load_uploaded_image(image_bytes)
    
    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    prediction = model.predict(img_array)
    
    # Interpretasi hasil
    if prediction[0][0] > 0.5:  # Asumsi threshold 0.5
        result = "Tumor detected"
    else:
        result = "No tumor detected"
    
    return img, result

# Halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek apakah file gambar diupload
        if 'image' not in request.files:
            return redirect(request.url)
        
        image_file = request.files['image']
        if image_file.filename == '':
            return redirect(request.url)
        
        # Baca file gambar
        image_bytes = image_file.read()
        
        # Prediksi hasil gambar
        img, result = predict_image(image_bytes)
        
        # Simpan gambar yang diupload
        upload_folder = 'static/uploads/'
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image_file.filename)
        img.save(image_path)
        
        # Tampilkan hasil prediksi
        return render_template('index.html', result=result, image_path=image_path)
    
    return render_template('index.html')

# Jalankan server Flask
if __name__ == '__main__':
    app.run(debug=True)
