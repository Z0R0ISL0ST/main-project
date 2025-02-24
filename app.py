from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model, label encoder, and scaler
model_filename = "maize_model_rf.pkl"
label_encoder_filename = "label_encoder.pkl"
scaler_filename = "scaler.pkl"

with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)
    
with open(label_encoder_filename, "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

with open(scaler_filename, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Image checking page
@app.route('/check_image', methods=['GET', 'POST'])
def check_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('check_image.html', message="No file selected!")

        file = request.files['image']
        if file.filename == '':
            return render_template('check_image.html', message="No file selected!")

        if file:
            fname=file.filename
            # Read and process the image
            image_path = os.path.join('static/images', file.filename)
            file.save(image_path)

            # Prediction logic
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
            image = image.flatten().reshape(1, -1)

            # Standardize image
            image = scaler.transform(image)

            # Predict class and probability
            probabilities = model.predict_proba(image)[0]
            predicted_class = model.predict(image)[0]
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            prediction_result = {
                "predicted_class": predicted_label,
                "probabilities": dict(zip(label_encoder.classes_, probabilities)),
            }

            return render_template('check_image.html', image_path=fname, prediction_result=prediction_result)

    return render_template('check_image.html')

if __name__ == '__main__':
    app.run(debug=True)
