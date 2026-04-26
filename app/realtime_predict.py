import cv2
import numpy as np
from tensorflow import keras
from PIL import Image, ImageEnhance, ImageFilter

# Load the trained model
model = keras.models.load_model("app/model.h5")

# The fixed label list (based on your training)
labels1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Start webcam
cap = cv2.VideoCapture(0)

# Prediction stabilization variables
last_prediction = None
last_confidence = 0
stabilization_counter = 0
STABILIZATION_THRESHOLD = 5

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Region of Interest (ROI)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    roi = frame[100:300, 100:300]

    # Convert to PIL image for better preprocessing
    image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Enhance sharpness
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)

    # Convert to grayscale
    image = image.convert('L')

    # Convert back to NumPy array
    img_np = np.array(image)

    # Histogram equalization
    img_np = cv2.equalizeHist(img_np)

    # Gaussian blur
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

    # Resize and normalize
    img_np = cv2.resize(img_np, (28, 28))
    img_np = img_np.astype('float32') / 255.0
    img_np = np.expand_dims(img_np, axis=-1)  # (28, 28, 1)
    img_np = np.expand_dims(img_np, axis=0)   # (1, 28, 28, 1)

    # Prediction
    pred = model.predict(img_np)
    current_label = np.argmax(pred)
    current_confidence = np.max(pred)
    current_char = labels1[current_label]

    # Display prediction
    color = (0, 255, 0) if current_confidence > 0.7 else (0, 0, 255)
    cv2.putText(frame, f"Prediction: {current_char}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Confidence: {current_confidence*100:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow("Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
