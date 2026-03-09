import tensorflow as tf
import numpy as np
import cv2

print("Starting script...", flush=True)

# Load model
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="multi_task_fruit_float16.tflite")
interpreter.allocate_tensors()
print("Model loaded.")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

fruit_labels = ["Apple","Banana","Grape","Guava","Jujube","Orange","Pomegranate","Strawberry"]

print("Loading image...")

img = cv2.imread("test_sp.jpeg")

if img is None:
    print("❌ Image NOT found. Check filename + location.")
    exit()

print("Image loaded.")

img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0).astype(np.float32)

print("Running inference...")

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

print("Inference done.")

fruit_out = interpreter.get_tensor(output_details[0]['index'])
fresh_out = interpreter.get_tensor(output_details[1]['index'])

fruit_idx = np.argmax(fruit_out)
fruit_name = fruit_labels[fruit_idx]

freshness_score = 1 / (1 + np.exp(-fresh_out[0][0]))

if freshness_score > 0.7:
    freshness = "Fresh"
elif freshness_score > 0.4:
    freshness = "Spoiling"
else:
    freshness = "Rotten"

print("✅ Predicted Fruit:", fruit_name)
print("✅ Freshness Score:", freshness_score)
print("✅ Freshness:", freshness)