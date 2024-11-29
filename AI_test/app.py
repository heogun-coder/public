from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# 저장된 모델 불러오기
model = tf.keras.models.load_model("git_test/AI_test/model/mnist_model.h5")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 이미지 데이터 받기
    image_data = request.get_json()["image"]
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # 이미지 전처리
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    # 예측
    prediction = model.predict(image)
    result = int(np.argmax(prediction[0]))

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
