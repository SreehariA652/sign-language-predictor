import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter
import time
import os

# Load model
model = load_model('app/model_v2.keras')

# Class indices
class_indices = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'del': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
    'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'nothing': 15, 'O': 16,
    'P': 17, 'Q': 18, 'R': 19, 'S': 20, 'space': 21, 'T': 22, 'U': 23, 'V': 24,
    'W': 25, 'X': 26, 'Y': 27, 'Z': 28
}
inv_class_indices = {v: k for k, v in class_indices.items()}

# Parameters
target_size = (64, 64)
confidence_threshold = 0.6
buffer_size = 5
predictions_buffer = deque(maxlen=buffer_size)

# Setup video capture
cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Tighter ROI: adjust these values if needed
    x1, y1, x2, y2 = 150, 100, 350, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]
    cv2.imshow("ROI", roi)

    # Preprocessing
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_eq = cv2.equalizeHist(roi_gray)
    roi_resized = cv2.resize(roi_eq, target_size)
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_rgb_like = np.stack([roi_normalized]*3, axis=-1)
    roi_expanded = np.expand_dims(roi_rgb_like, axis=0)

    # Save debug ROI if needed
    cv2.imwrite("debug_roi.jpg", roi)

    # Prediction
    pred_probs = model.predict(roi_expanded)[0]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class]
    predicted_label = inv_class_indices[pred_class]

    if confidence > confidence_threshold:
        predictions_buffer.append(pred_class)
    else:
        predictions_buffer.append(None)

    valid_preds = [p for p in predictions_buffer if p is not None]
    if valid_preds:
        most_common_pred, count = Counter(valid_preds).most_common(1)[0]
        predicted_label = inv_class_indices[most_common_pred]
        display_text = f"{predicted_label} ({confidence * 100:.1f}%)"
    else:
        display_text = "No confident prediction"

    # Display
    cv2.putText(frame, display_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Real-Time Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
