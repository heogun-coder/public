from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import io
import numpy as np
from image_blender import FaceBlender
import traceback

app = Flask(__name__)
face_blender = FaceBlender()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/blend", methods=["POST"])
def blend():
    try:
        # 파일 존재 여부 확인
        if "image1" not in request.files or "image2" not in request.files:
            return jsonify({"error": "두 개의 이미지가 필요합니다."}), 400

        image1_file = request.files["image1"]
        image2_file = request.files["image2"]

        # 파일 유효성 검사
        if image1_file.filename == "" or image2_file.filename == "":
            return jsonify({"error": "이미지가 선택되지 않았습니다."}), 400

        # 알파값 검증
        try:
            alpha = float(request.form.get("alpha", 0.5))
            if not 0 <= alpha <= 1:
                raise ValueError
        except ValueError:
            return jsonify({"error": "유효하지 않은 알파값입니다."}), 400

        # 이미지 형식 검증 및 변환
        try:
            image1 = Image.open(image1_file).convert("RGB")
            image2 = Image.open(image2_file).convert("RGB")
        except Exception:
            return jsonify({"error": "유효하지 않은 이미지 형식입니다."}), 400

        # 이미지 블렌딩
        result = face_blender.process_images(image1, image2, alpha)

        if result is None:
            return (
                jsonify(
                    {
                        "error": "얼굴을 찾을 수 없거나 이미지 처리 중 오류가 발생했습니다."
                    }
                ),
                400,
            )

        # 결과 이미지 변환 및 전송
        try:
            result_image = Image.fromarray(result)
            img_io = io.BytesIO()
            result_image.save(img_io, "PNG")
            img_io.seek(0)
            return send_file(img_io, mimetype="image/png")
        except Exception:
            return jsonify({"error": "결과 이미지 생성 중 오류가 발생했습니다."}), 500

    except Exception as e:
        print("Error details:", traceback.format_exc())  # 서버 로그에 상세 에러 출력
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
