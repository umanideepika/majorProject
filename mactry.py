import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
print("Script started")

# Labels (same order as training)
labels = [
    "FreshApple",
    "FreshBanana",
    "FreshGrape",
    "FreshGuava",
    "FreshJujube",
    "FreshOrange",
    "FreshPomegranate",
    "FreshStrawberry",
    "RottenApple",
    "RottenBanana",
    "RottenGrape",
    "RottenGuava",
    "RottenJujube",
    "RottenOrange",
    "RottenPomegranate",
    "RottenStrawberry"
]

# Load model
interpreter = Interpreter(model_path="multi_task_fruit_float16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open Mac webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected")
    exit()

print("Mac TFLite test running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224,224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)
    conf = np.max(output)

    text = f"{labels[pred]} ({conf*100:.1f}%)"

    cv2.putText(frame, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("Mac TFLite Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()