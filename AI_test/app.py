from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import keras
from tensorflow.python.keras import img_to_array
import cv2

app = Flask(__name__)

# 저장된 모델 불러오기
model = tf.keras.models.load_model("git_test/AI_test/model/mnist_model.h5")


# 이미지 전처리 함수 추가
def preprocess_image(image):
    # 노이즈 제거
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 이미지 이진화
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 테두리 강화
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    return image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 이미지 데이터 받기
    image_data = request.get_json()["image"]
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # 향상된 이미지 전처리
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    image = preprocess_image(image)

    # 모델 입력을 위한 전처리
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    # 앙상블 예측 (여러 번 예측하여 평균)
    predictions = []
    for _ in range(5):
        pred = model.predict(image)
        predictions.append(pred[0])

    # 평균 예측값 계산
    avg_prediction = np.mean(predictions, axis=0)
    result = int(np.argmax(avg_prediction))
    confidence = float(avg_prediction[result])

    return jsonify({"prediction": result, "confidence": round(confidence * 100, 2)})


if __name__ == "__main__":
    app.run(debug=True)
