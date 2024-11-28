from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import io
import cv2
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


@app.route("/blend_video", methods=["POST"])
def blend_video():
    try:
        # 비디오 파일 받기
        video1_file = request.files["video1"]
        video2_file = request.files["video2"]
        alpha = float(request.form["alpha"])

        # 임시 파일로 저장
        temp_path1 = "temp_video1.mp4"
        temp_path2 = "temp_video2.mp4"
        video1_file.save(temp_path1)
        video2_file.save(temp_path2)

        # 비디오 캡처 객체 생성
        cap1 = cv2.VideoCapture(temp_path1)
        cap2 = cv2.VideoCapture(temp_path2)

        # 결과 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "result_video.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # OpenCV BGR을 RGB로 변환
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # 각 프레임 블렌딩
            result_frame = face_blender.process_images(
                Image.fromarray(frame1_rgb), Image.fromarray(frame2_rgb), alpha
            )

            if result_frame is not None:
                # RGB를 BGR로 다시 변환
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                out.write(result_frame_bgr)

        # 리소스 해제
        cap1.release()
        cap2.release()
        out.release()

        # 결과 비디오 전송
        return send_file(output_path, mimetype="video/mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
