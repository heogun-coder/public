from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
import dlib
import os
import tempfile
from werkzeug.utils import secure_filename
import yt_dlp
import logging
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB 제한
socketio = SocketIO(app)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
)


def download_youtube(url, output_path):
    ydl_opts = {"format": "best[ext=mp4]", "outtmpl": output_path, "quiet": False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def face_swap_frame(source_frame, target_frame):
    # 그레이스케일 변환
    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    source_faces = detector(source_gray)
    target_faces = detector(target_gray)

    if len(source_faces) == 0 or len(target_faces) == 0:
        return target_frame

    # 첫 번째 얼굴에 대해서만 처리
    source_face = source_faces[0]
    target_face = target_faces[0]

    # 랜드마크 추출
    source_landmarks = predictor(source_gray, source_face)
    target_landmarks = predictor(target_gray, target_face)

    # 랜드마크 포인트 추출
    source_points = np.array([[p.x, p.y] for p in source_landmarks.parts()])
    target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

    # 얼굴 영역 마스크 생성
    hull = cv2.convexHull(target_points)
    mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # 워핑을 위한 변환 행렬 계산
    M = cv2.estimateAffinePartial2D(source_points, target_points)[0]

    # 소스 이미지 워핑
    warped = cv2.warpAffine(
        source_frame, M, (target_frame.shape[1], target_frame.shape[0])
    )

    # 포아송 블렌딩
    center = (target_face.center().x, target_face.center().y)
    output = cv2.seamlessClone(warped, target_frame, mask, center, cv2.NORMAL_CLONE)

    return output


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_videos", methods=["POST"])
def process_videos():
    try:
        # YouTube URL 또는 파일 받기
        source_type = request.form.get("source_type")
        target_type = request.form.get("target_type")

        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, "source.mp4")
        target_path = os.path.join(temp_dir, "target.mp4")

        # 소스 비디오 처리
        if source_type == "youtube":
            source_url = request.form.get("source_url")
            download_youtube(source_url, source_path)
        else:
            source_file = request.files["source_file"]
            source_file.save(source_path)

        # 타겟 비디오 처리
        if target_type == "youtube":
            target_url = request.form.get("target_url")
            download_youtube(target_url, target_path)
        else:
            target_file = request.files["target_file"]
            target_file.save(target_path)

        # 비디오 처리
        source_cap = cv2.VideoCapture(source_path)
        target_cap = cv2.VideoCapture(target_path)

        # 출력 비디오 설정
        fps = int(target_cap.get(cv2.CAP_PROP_FPS))
        width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(temp_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = int(target_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while True:
            ret1, source_frame = source_cap.read()
            ret2, target_frame = target_cap.read()

            if not ret1 or not ret2:
                break

            current_frame += 1
            progress = int((current_frame / frame_count) * 100)

            # 프레임 처리
            output_frame = face_swap_frame(source_frame, target_frame)
            out.write(output_frame)

            # 진행률 업데이트
            if current_frame % 10 == 0:
                socketio.emit(
                    "progress_update",
                    {
                        "progress": progress,
                        "message": f"프레임 처리 중... ({current_frame}/{frame_count})",
                    },
                )

        source_cap.release()
        target_cap.release()
        out.release()

        return send_file(output_path, mimetype="video/mp4")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
