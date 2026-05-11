import tensorflow as tf
import numpy as np

# Load saved model
model = tf.keras.models.load_model(
    'workspace/cancer-pred/model/saved_models/cancer-pred.h5')

# Sample input - malignant case (from Wisconsin dataset)
sample = np.array([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189]])

prediction = model.predict(sample)
prob = prediction[0][0]
result = "Malignant" if prob > 0.5 else "Benign"
print(f"Prediction: {result} (confidence: {prob:.4f})")
