from flask import Flask, request, jsonify, render_template, redirect, url_for
from utils import load_model, load_classes, predict_image_from_bytes
import base64

app = Flask(__name__)

# Load model and class names once at startup
model = load_model()
class_names = load_classes()

# ---------- Routes for Pages ----------
@app.route('/')
def home():
    return render_template('home.html')  # Main page with 3 sections

@app.route('/live')
def live_prediction():
    return render_template('index.html')  # Live webcam prediction page

@app.route('/upload')
def upload_page():
    return render_template('upload.html')  # Upload prediction page

@app.route('/voice')
def voice_input():
    return render_template('voice.html')  # Voice input page (placeholder)

# ---------- Live Prediction API ----------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    try:
        # Decode the base64-encoded image from webcam
        image_data = base64.b64decode(data['image'].split(',')[1])
        prediction = predict_image_from_bytes(image_data, model, class_names)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------- Upload Image Prediction API ----------
@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'image' not in request.files:
        return redirect(url_for('upload_page'))

    image = request.files['image']
    if image.filename == '':
        return redirect(url_for('upload_page'))

    try:
        image_bytes = image.read()
        prediction = predict_image_from_bytes(image_bytes, model, class_names)
        return render_template('upload.html', prediction=prediction)
    except Exception as e:
        return render_template('upload.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
