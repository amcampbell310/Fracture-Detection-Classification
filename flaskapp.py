from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from base64 import b64encode
import torch
# Load the saved model
loaded_model = load_model("")
object_detection_model = torch.load("")

app = Flask(__name__)

def preprocess_image(image):
    # Resize the image to match model input size
    image = cv2.resize(image, (454, 373))
    # Normalize pixel values
    image = image / 255.0
    return image

def predict_fracture(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = loaded_model.predict(np.expand_dims(processed_image, axis=0))
    # Convert prediction to 'fractured' or 'non_fractured'
    if prediction[0][0] > 0.5:
        prediction_label = 'fractured'
    else:
        prediction_label = 'non_fractured'
    # Prediction probability
    prediction_prob = prediction[0][0]
    return prediction_label, prediction_prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Read the image file
        img = Image.open(uploaded_file)
        # Convert image to numpy array
        img_array = np.array(img)
        # Get prediction result
        prediction = predict_fracture(img_array)
        # Get prediction probability
        prediction_probability = loaded_model.predict(np.expand_dims(preprocess_image(img_array), axis=0))[0][0]
        # Encode image as base64 string
        img_str = BytesIO()
        img.save(img_str, format='JPEG')
        img_base64 = str(b64encode(img_str.getvalue()))[2:-1]
        # Render the template with prediction result
        return render_template('index.html', prediction=prediction, prediction_probability=prediction_probability, img_base64=img_base64)
    else:
        # If no file is uploaded, render the template without prediction result
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
