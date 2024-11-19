from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import io
import numpy as np
from image_blender import FaceBlender

app = Flask(__name__)
face_blender = FaceBlender()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blend", methods=["POST"])
def blend():
    try:
        # 이미지 파일과 알파값 받기
        image1_file = request.files["image1"]
        image2_file = request.files["image2"]
        alpha = float(request.form["alpha"])

        # 이미지 검증
        if not image1_file or not image2_file:
            return jsonify({"error": "두 개의 이미지가 필요합니다."}), 400

        # PIL Image로 변환
        image1 = Image.open(image1_file).convert("RGB")
        image2 = Image.open(image2_file).convert("RGB")

        # 이미지 블렌딩
        result = face_blender.process_images(image1, image2, alpha)

        if result is None:
            return jsonify({"error": "이미지 블렌딩 실패"}), 400

        # numpy 배열을 PIL Image로 변환
        result_image = Image.fromarray(result)

        # 결과 이미지를 바이트 스트림으로 변환
        img_io = io.BytesIO()
        result_image.save(img_io, "PNG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
