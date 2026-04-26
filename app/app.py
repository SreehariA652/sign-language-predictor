from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import io

app = Flask(__name__)
model = keras.models.load_model('app/model.h5')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Apply sharpening
    image = image.filter(ImageFilter.SHARPEN)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Convert to grayscale
    image = image.convert('L')

    # Resize
    image = image.resize((28, 28))

    # Normalize and reshape
    img_arr = np.array(image).astype('float32') / 255
    img_arr = np.expand_dims(img_arr, axis=-1)  # Add channel dim (28,28,1)
    img_arr = np.expand_dims(img_arr, axis=0)   # Add batch dim (1,28,28,1)

    return img_arr

@app.route('/')
def index():
    return render_template('index.html', prediction=None) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded.")

    image_file = request.files['image']
    image_bytes = image_file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_index = int(np.argmax(prediction))
    predicted_label = labels[predicted_index]

    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
