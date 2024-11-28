import pandas as pd
import pickle #menampilkan yg sdh kt buat

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Mengaktifkan CORS untuk mengizinkan semua domain
CORS(app)

# Membuat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return "<h1>Selamat Datang di API DS Model</h1>"

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    # Gunakan try catch untuk menampilkan status gangguan atau error
    try:
        #ambil data request
        data = request.get_json()

        #Input untuk memprediksi diabetes
        # prediksi diabetes berdasarkan faktor-faktor
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        # Melakukan prediksi
        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        # Probabilitas positif dan negatif dalam bentuk persentase
        probability_negative = probabilities[0][0] * 100 #persentase untuk kelas 0 negatif
        probability_positive = probabilities[0][1] * 100 #persentase untuk kelas 1 negatif
        
        # prediksi output (untuk 1 itu positif diabetes)
        if prediction[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami, kemungkinan menderita diabetes adalah {probability_positive:.2f}%'
        else:
            result = "HAsil prediksi menunjukkan Anda kemungkinan rendah diabetes"
        
        # menampilkan hasil prediksi dan probabilitas dalam bentuk JSON
        return jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",
                'positive': f"{probability_positive:.2f}%"
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)