from flask import Flask, render_template, request, send_file, jsonify, make_response
import cv2
import numpy as np
import dlib
import os
import tempfile
from werkzeug.utils import secure_filename
import logging
from flask_socketio import SocketIO
import yt_dlp
import re
import time


# 프로젝트 설정 함수
def setup_project():
    # templates 폴더 확인
    if not os.path.exists("templates"):
        os.makedirs("templates")

    # static 폴더 확인
    if not os.path.exists("static"):
        os.makedirs("static")

    # dlib 모델 파일 확인
    if not os.path.exists(
        "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
    ):
        logger.error("얼굴 랜드마크 모델 파일이 없습니다!")


# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 프로젝트 설정 실행
setup_project()

# Flask 앱 초기화
app = Flask(__name__)
socketio = SocketIO(app)

# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
)


def face_swap_frame(source_frame, target_frame):
    try:
        # 프레임 복사
        source_frame = source_frame.copy()
        target_frame = target_frame.copy()

        # 그레이스케일 변환 및 히스토그램 평활화
        source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        source_gray = cv2.equalizeHist(source_gray)
        target_gray = cv2.equalizeHist(target_gray)

        # 다양한 스케일로 얼굴 검출 시도
        scales = [0.5, 1.0, 1.5]
        source_faces = []
        target_faces = []

        for scale in scales:
            if len(source_faces) == 0:
                scaled_source = cv2.resize(source_gray, None, fx=scale, fy=scale)
                source_faces = detector(scaled_source)

            if len(target_faces) == 0:
                scaled_target = cv2.resize(target_gray, None, fx=scale, fy=scale)
                target_faces = detector(scaled_target)

            if len(source_faces) > 0 and len(target_faces) > 0:
                break

        if len(source_faces) == 0 or len(target_faces) == 0:
            return None

        # 가장 큰 얼굴 선택
        source_face = max(source_faces, key=lambda rect: rect.area())
        target_face = max(target_faces, key=lambda rect: rect.area())

        # 랜드마크 추출
        source_landmarks = predictor(source_gray, source_face)
        target_landmarks = predictor(target_gray, target_face)

        # 포인트 추출
        source_points = np.array([[p.x, p.y] for p in source_landmarks.parts()])
        target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

        # 변환 행렬 계산
        M = cv2.estimateAffinePartial2D(source_points, target_points)[0]

        # 마스크 생성
        hull = cv2.convexHull(target_points)
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # 워핑 및 블렌딩
        warped = cv2.warpAffine(
            source_frame, M, (target_frame.shape[1], target_frame.shape[0])
        )
        output = cv2.seamlessClone(
            warped,
            target_frame,
            mask,
            (target_face.center().x, target_face.center().y),
            cv2.NORMAL_CLONE,
        )

        return output

    except Exception as e:
        logger.error(f"얼굴 스왑 중 오류: {str(e)}")
        return None


def process_video(source_path, target_path, output_path):
    try:
        source_cap = cv2.VideoCapture(source_path)
        target_cap = cv2.VideoCapture(target_path)
        out = None

        if not source_cap.isOpened() or not target_cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없습니다")

        # 비디오 속성
        fps = int(target_cap.get(cv2.CAP_PROP_FPS))
        width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(target_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(fps * 20, total_frames)  # 20초 제한

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 소스 프레임 읽기
        ret, source_frame = source_cap.read()
        if not ret:
            raise ValueError("소스 비디오를 읽을 수 없습니다")

        current_frame = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # 연속 실패 허용 횟수

        while current_frame < max_frames:
            ret, target_frame = target_cap.read()
            if not ret:
                break

            try:
                output_frame = face_swap_frame(source_frame, target_frame)

                if output_frame is not None:
                    consecutive_failures = 0  # 성공 시 카운터 리셋
                    out.write(output_frame)
                else:
                    consecutive_failures += 1
                    out.write(target_frame)

                    if consecutive_failures > max_consecutive_failures:
                        logger.warning("연속 실패 횟수 초과, 처리 중단")
                        break

            except Exception as e:
                logger.error(f"프레임 {current_frame} 처리 실패: {str(e)}")
                out.write(target_frame)

            current_frame += 1
            if current_frame % 5 == 0:
                progress = int((current_frame / max_frames) * 100)
                socketio.emit(
                    "progress_update",
                    {
                        "progress": progress,
                        "message": f"처리 중... ({current_frame}/{max_frames})",
                    },
                )

        # 리소스 해제
        if "source_cap" in locals() and source_cap is not None:
            source_cap.release()
        if "target_cap" in locals() and target_cap is not None:
            target_cap.release()
        if "out" in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()  # 열려있는 모든 창 닫기

        return True

    except Exception as e:
        logger.error(f"비디오 처리 중 오류: {str(e)}")
        raise
    finally:
        # 리소스 해제 보장
        if "source_cap" in locals():
            source_cap.release()
        if "target_cap" in locals():
            target_cap.release()
        if "out" in locals():
            out.release()


def download_youtube(url, output_path):
    try:

        def progress_hook(d):
            if d["status"] == "downloading":
                # 다운로드 진행률 계산
                total_bytes = d.get("total_bytes")
                downloaded_bytes = d.get("downloaded_bytes", 0)
                if total_bytes:
                    progress = (downloaded_bytes / total_bytes) * 100
                    socketio.emit(
                        "progress_update",
                        {
                            "progress": int(progress),
                            "message": f"YouTube 다운로드 중... {int(progress)}%",
                        },
                    )

        ydl_opts = {
            "format": "best[ext=mp4]",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("다운로드 시작...")
            ydl.download([url])
            logger.info("다운로드 완료")

        return True

    except Exception as e:
        logger.error(f"YouTube 다운로드 중 오류: {str(e)}")
        raise ValueError(f"YouTube 다운로드 실패: {str(e)}")


def validate_youtube_url(url):
    patterns = [
        r"^(https?:\/\/)?(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[a-zA-Z0-9_-]{11}",
        r"^(https?:\/\/)?(www\.)?youtube\.com\/shorts\/[a-zA-Z0-9_-]{11}",
        r"^(https?:\/\/)?(www\.)?youtube\.com\/embed\/[a-zA-Z0-9_-]{11}",
    ]

    return any(re.match(pattern, url) for pattern in patterns)


# 메인 페이지 라우트 추가
@app.route("/")
def index():
    return render_template("index.html")


# 비디오 처리 라우트
@app.route("/process_videos", methods=["POST"])
def process_videos():
    temp_dir = None
    response = None

    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, "source.mp4")
        target_path = os.path.join(temp_dir, "target.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")

        # 입력 타입 확인
        source_type = request.form.get("source_type")
        target_type = request.form.get("target_type")

        logger.info(f"소스 타입: {source_type}, 타겟 타입: {target_type}")

        # 소스 비디오 처리
        if source_type == "file":
            if "source_file" not in request.files:
                raise ValueError("소스 파일이 전송되지 않았습니다")
            source_file = request.files["source_file"]
            if source_file.filename == "":
                raise ValueError("소스 파일이 선택되지 않았습니다")
            source_file.save(source_path)
            logger.info("소스 파일 저장 완료")
        elif source_type == "youtube":
            source_url = request.form.get("source_url")
            if not source_url:
                raise ValueError("YouTube URL이 제공되지 않았습니다")
            if not validate_youtube_url(source_url):
                raise ValueError("유효하지 않은 YouTube URL입니다")
            download_youtube(source_url, source_path)
            logger.info("소스 YouTube 다운로드 완료")

        # 타겟 비디오 처리
        if target_type == "file":
            if "target_file" not in request.files:
                raise ValueError("타겟 파일이 전송되지 않았습니다")
            target_file = request.files["target_file"]
            if target_file.filename == "":
                raise ValueError("타겟 파일이 선택되지 않았습니다")
            target_file.save(target_path)
            logger.info("타겟 파일 저장 완료")
        elif target_type == "youtube":
            target_url = request.form.get("target_url")
            if not target_url:
                raise ValueError("YouTube URL이 제공되지 않았습니다")
            if not validate_youtube_url(target_url):
                raise ValueError("유효하지 않은 YouTube URL입니다")
            download_youtube(target_url, target_path)
            logger.info("타겟 YouTube 다운로드 완료")

        # 비디오 처리
        logger.info("비디오 처리 시작")
        process_video(source_path, target_path, output_path)

        if not os.path.exists(output_path):
            raise ValueError("출력 파일이 생성되지 않았습니다")

        # 파일을 메모리에 복사하고 응답 생성
        with open(output_path, "rb") as f:
            file_content = f.read()

        response = make_response(file_content)
        response.headers["Content-Type"] = "video/mp4"
        response.headers["Content-Disposition"] = "attachment; filename=output.mp4"

        return response

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 400

    finally:
        # 안전한 임시 파일 정리
        if temp_dir and os.path.exists(temp_dir):
            try:
                cleanup_temp_directory(temp_dir)
            except Exception as e:
                logger.error(f"임시 파일 정리 중 오류: {str(e)}")


def cleanup_temp_directory(temp_dir):
    """
    임시 디렉토리와 그 내용을 안전하게 정리
    """
    max_attempts = 3
    delay = 1  # 초

    for attempt in range(max_attempts):
        try:
            # 디렉토리 내의 모든 파일을 순회
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                # 먼저 파일 삭제
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        if os.path.exists(file_path):
                            os.chmod(file_path, 0o777)  # 파일 권한 변경
                            os.unlink(file_path)
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패 {file_path}: {str(e)}")

                # 그 다음 디렉토리 삭제
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        if os.path.exists(dir_path):
                            os.chmod(dir_path, 0o777)  # 디렉토리 권한 변경
                            os.rmdir(dir_path)
                    except Exception as e:
                        logger.warning(f"디렉토리 삭제 실패 {dir_path}: {str(e)}")

            # 마지막으로 루트 디렉토리 삭제
            if os.path.exists(temp_dir):
                os.chmod(temp_dir, 0o777)
                os.rmdir(temp_dir)

            break  # 성공적으로 삭제되었다면 루프 종료

        except Exception as e:
            if attempt < max_attempts - 1:  # 마지막 시도가 아니라면
                logger.warning(f"임시 파일 정리 재시도 {attempt + 1}/{max_attempts}")
                time.sleep(delay)  # 잠시 대기 후 재시도
            else:
                logger.error(f"임시 파일 정리 최종 실패: {str(e)}")
                raise


if __name__ == "__main__":
    socketio.run(app, debug=True)
