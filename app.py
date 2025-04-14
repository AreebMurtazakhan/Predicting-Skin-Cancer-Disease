from flask import Flask, render_template, request
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploaded_images')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
loaded_model = load_model("model.keras")
img_size = 224

# Dictionary containing disease details
disease_info = {
    'Actinic Keratosis': {
        'Salts': ['Fluorouracil', 'Imiquimod', 'Diclofenac'],
        'Medicines': ['Efudex Cream', 'Aldara Cream', 'Solaraze Gel'],
        'Precautions': [
            'Avoid prolonged sun exposure', 
            'Use broad-spectrum sunscreen SPF 30+', 
            'Wear protective clothing'
        ],
        'Recommendations': [
            'Regular dermatological checkups', 
            'Maintain skin hydration', 
            'Avoid tanning beds'
        ]
    },
    'Basal Cell Carcinoma': {
        'Salts': ['Vismodegib', 'Sonidegib', 'Fluorouracil'],
        'Medicines': ['Erivedge Capsules', 'Odomzo Capsules', 'Efudex Cream'],
        'Precautions': [
            'Protect skin from UV rays', 
            'Apply sunscreen daily', 
            'Avoid smoking'
        ],
        'Recommendations': [
            'Early detection through skin screenings', 
            'Follow post-surgery care guidelines', 
            'Healthy diet with antioxidants'
        ]
    },
    'Melanoma': {
        'Salts': ['Pembrolizumab', 'Nivolumab', 'Dabrafenib'],
        'Medicines': ['Keytruda Injection', 'Opdivo Injection', 'Tafinlar Capsules'],
        'Precautions': [
            'Limit sun exposure', 
            'Avoid tanning beds', 
            'Use SPF 50+ sunscreen'
        ],
        'Recommendations': [
            'Routine skin self-exams', 
            'Consult an oncologist if changes in moles occur', 
            'Maintain a healthy immune system'
        ]
    },
    'Nevus': {
        'Salts': ['None (typically not treated unless atypical)'],
        'Medicines': ['Topical retinoids (if needed)', 'Surgical excision if atypical'],
        'Precautions': [
            'Monitor changes in size, shape, or color', 
            'Use sunscreen daily', 
            'Avoid UV exposure'
        ],
        'Recommendations': [
            'Annual skin exams', 
            'Avoid irritating the nevus', 
            'Seek medical advice if itching or bleeding occurs'
        ]
    },
    'Pigmented Benign Keratosis': {
        'Salts': ['Cryotherapy agents', 'Trichloroacetic Acid (TCA)'],
        'Medicines': ['Liquid Nitrogen Spray', 'TCA Solution', 'Laser therapy if needed'],
        'Precautions': [
            'Protect from excessive sun exposure', 
            'Avoid scratching or picking the lesion', 
            'Keep skin moisturized'
        ],
        'Recommendations': [
            'Maintain skin hygiene', 
            'Use mild skincare products', 
            'Regular checkups for abnormal changes'
        ]
    },
    'Seborrheic Keratosis': {
        'Salts': ['Hydrogen Peroxide', 'Cryotherapy agents', 'Salicylic Acid'],
        'Medicines': ['Eskata Solution', 'Liquid Nitrogen Spray', 'Salicylic Acid Cream'],
        'Precautions': [
            'Avoid excessive sun exposure', 
            'Do not scratch or rub lesions', 
            'Use gentle skincare products'
        ],
        'Recommendations': [
            'Regular dermatological evaluations', 
            'Maintain skin hydration', 
            'Monitor for any unusual changes'
        ]
    }
}
# List of disease names (order must match your model's output)
disease_names = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma', 'Nevus', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")
    # Convert BGR to RGB, resize, and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img / 255.0

def predict_single_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    class_label = disease_names[predicted_class]
    additional_info = disease_info.get(class_label, {})
    return class_label, confidence, additional_info

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None
    confidence = None
    additional_info = {}
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
            file.save(save_path)
            predicted_label, confidence, additional_info = predict_single_image(save_path)
            image_path = '/' + save_path.replace("\\", "/")
    return render_template("index.html", predicted_label=predicted_label, 
                           confidence=confidence, additional_info=additional_info, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
