from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)

# Load the pre-trained machine learning model
model = keras.models.load_model('./model/apple_classification_model.h5')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data gambar yang diunggah dari form
        image = request.files['image']

        # Membaca data gambar dan mengonversinya ke format base64
        image_data =   image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Prapemrosesan gambar: membuka, mengubah ukuran, dan mengonversi ke array numpy
        img = Image.open(image)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0  # Normalisasi nilai piksel menjadi 0-1
        img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch

        # Lakukan prediksi dengan model
        prediction = model.predict(img_array)

        # Menentukan apakah buah tersebut busuk atau tidak berdasarkan hasil prediksi
        if prediction[0][0] >= 0.5:
            hasil_prediksi = "Busuk"
        else:
            hasil_prediksi = "Segar"

        # Mengirim hasil prediksi dan data gambar ke template index.html
        return render_template('index.html', hasil_prediksi=hasil_prediksi, image_base64=image_base64)
    except Exception as e:
        return str(e)

if __name__ == '_main_':
    app.run(debug=True)