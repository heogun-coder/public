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
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("video_processor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# 임시 파일 관리를 위한 함수
def cleanup_temp_files(*paths):
    """
    임시 파일들을 안전하게 삭제하는 함수
    """
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    os.rmdir(path)
                logger.debug("파일/디렉토리 삭제: %s", path)
        except Exception as e:
            logger.error("cleanup 중 오류 발생: %s", str(e))


# dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
)

# OpenCV 설정
cv2.setNumThreads(4)  # 스레드 수 제한


def download_youtube(url, output_path):
    ydl_opts = {"format": "best[ext=mp4]", "outtmpl": output_path, "quiet": False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def face_swap_frame(source_frame, target_frame):
    try:
        # 프레임 복사 및 크기 조정
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
                scaled_source = cv2.resize(
                    source_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                source_faces = detector(scaled_source)

            if len(target_faces) == 0:
                scaled_target = cv2.resize(
                    target_gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                target_faces = detector(scaled_target)

            if len(source_faces) > 0 and len(target_faces) > 0:
                break

        if len(source_faces) == 0 or len(target_faces) == 0:
            logger.debug("얼굴 감지 실패 (스케일 조정 후에도)")
            return None

        # 가장 큰 얼굴 선택
        source_face = max(source_faces, key=lambda rect: rect.area())
        target_face = max(target_faces, key=lambda rect: rect.area())

        # 랜드마크 추출 전 이미지 전처리
        source_gray = cv2.GaussianBlur(source_gray, (3, 3), 0)
        target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)

        # 랜드마크 추출
        source_landmarks = predictor(source_gray, source_face)
        target_landmarks = predictor(target_gray, target_face)

        # 얼굴 영역 계산
        source_points = np.array([[p.x, p.y] for p in source_landmarks.parts()])
        target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

        # 변환 행렬 계산
        M = cv2.estimateAffinePartial2D(source_points, target_points)[0]

        # 마스크 생성
        hull = cv2.convexHull(target_points)
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # 소스 이미지 워핑
        warped = cv2.warpAffine(
            source_frame,
            M,
            (target_frame.shape[1], target_frame.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 포아송 블렌딩
        center = (target_face.center().x, target_face.center().y)
        output = cv2.seamlessClone(warped, target_frame, mask, center, cv2.NORMAL_CLONE)

        return output

    except Exception as e:
        logger.error(f"얼굴 스왑 중 오류: {str(e)}")
        return None


def process_video(source_path, target_path, output_path):
    source_cap = None
    target_cap = None
    out = None
    temp_output = output_path.replace(".mp4", "_temp.mp4")

    try:
        source_cap = cv2.VideoCapture(source_path)
        target_cap = cv2.VideoCapture(target_path)

        if not source_cap.isOpened() or not target_cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없습니다.")

        # 비디오 속성
        fps = int(target_cap.get(cv2.CAP_PROP_FPS))
        width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(target_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"비디오 속성 - FPS: {fps}, 크기: {width}x{height}, 총 프레임: {total_frames}"
        )

        # mp4v 코덱 사용
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height), isColor=True)

        if not out.isOpened():
            raise ValueError("출력 비디오 파일을 생성할 수 없습니다.")

        # 소스 비디오의 첫 프레임에서 얼굴 찾기
        ret, source_frame = source_cap.read()
        if not ret:
            raise ValueError("소스 비디오를 읽을 수 없습니다.")

        source_face_frame = source_frame.copy()
        current_frame = 0

        logger.info("프레임 처리 시작")

        while True:
            ret2, target_frame = target_cap.read()
            if not ret2:
                break

            current_frame += 1

            try:
                # 타겟 프레임 크기 확인 및 조정
                if target_frame.shape[:2] != (height, width):
                    target_frame = cv2.resize(target_frame, (width, height))

                # 얼굴 스왑 시도
                output_frame = face_swap_frame(source_face_frame, target_frame)

                # 얼굴이 감지되지 않은 경우 원본 프레임 사용
                if output_frame is None:
                    logger.debug(f"프레임 {current_frame}: 얼굴이 감지되지 않음")
                    output_frame = target_frame

                # 프레임 저장
                out.write(output_frame)

                # 진행률 업데이트 (10프레임마다)
                if current_frame % 10 == 0:
                    progress = int((current_frame / total_frames) * 100)
                    logger.debug(f"처리 진행률: {progress}%")
                    socketio.emit(
                        "progress_update",
                        {
                            "progress": progress,
                            "message": f"프레임 처리 중... ({current_frame}/{total_frames})",
                        },
                    )

            except Exception as e:
                logger.warning(f"프레임 {current_frame} 처리 실패: {str(e)}")
                # 오류 발생 시 원본 프레임 사용
                out.write(target_frame)

        logger.info(f"총 {current_frame}개 프레임 처리 완료")

        # 리소스 해제
        source_cap.release()
        target_cap.release()
        out.release()

        # FFmpeg로 최종 변환 시도
        try:
            import subprocess

            logger.info("FFmpeg로 비디오 변환 시작")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    temp_output,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    output_path,
                ],
                check=True,
            )
            os.remove(temp_output)
            logger.info("FFmpeg 변환 완료")
        except Exception as e:
            logger.warning(f"FFmpeg 변환 실패, 원본 파일 사용: {str(e)}")
            os.rename(temp_output, output_path)

        return True

    except Exception as e:
        logger.error(f"비디오 처리 중 오류 발생: {str(e)}")
        raise

    finally:
        # 리소스 정리
        if source_cap is not None:
            source_cap.release()
        if target_cap is not None:
            target_cap.release()
        if out is not None:
            out.release()
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception as e:
                logger.error(f"임시 파일 삭제 실패: {str(e)}")


def enhance_video(video_path, output_path, duration=20):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("비디오를 열 수 없습니다.")

        # 비디오 속성
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 해상도 증가 (1.5배로 조정)
        new_width = int(width * 1.5)
        new_height = int(height * 1.5)

        # 20초까지만 처리
        max_frames = min(int(duration * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # 코덱 변경
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # avc1 대신 mp4v 사용
        temp_output = output_path.replace(".mp4", "_temp.mp4")

        out = cv2.VideoWriter(temp_output, fourcc, fps, (new_width, new_height))

        if not out.isOpened():
            raise ValueError("출력 비디오 파일을 생성할 수 없습니다.")

        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 화질 개선
            enhanced_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )

            # 노이즈 제거 (파라미터 조정)
            enhanced_frame = cv2.fastNlMeansDenoisingColored(
                enhanced_frame,
                None,
                h=5,  # 색상 필터링 강도 감소
                hColor=5,  # 색상 구성 요소 필터링 강도 감소
                templateWindowSize=7,
                searchWindowSize=21,
            )

            out.write(enhanced_frame)
            frame_count += 1

        # 리소스 해제
        cap.release()
        out.release()

        # FFmpeg로 최종 변환 (if available)
        try:
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    temp_output,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    output_path,
                ],
                check=True,
            )
            os.remove(temp_output)
        except:
            # FFmpeg를 사용할 수 없는 경우 임시 파일을 최종 파일로 사용
            os.rename(temp_output, output_path)

        return True

    except Exception as e:
        logger.error(f"비디오 화질 향상 중 오류: {str(e)}")
        if "temp_output" in locals() and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        raise


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_videos", methods=["POST"])
def process_videos():
    temp_dir = None
    source_path = None
    target_path = None
    output_path = None
    enhanced_source = None
    enhanced_target = None

    try:
        # YouTube URL 또는 파일 받기
        source_type = request.form.get("source_type")
        target_type = request.form.get("target_type")

        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, "source.mp4")
        target_path = os.path.join(temp_dir, "target.mp4")
        enhanced_source = os.path.join(temp_dir, "enhanced_source.mp4")
        enhanced_target = os.path.join(temp_dir, "enhanced_target.mp4")
        output_path = os.path.join(temp_dir, "output.mp4")

        logger.info("임시 디렉토리 생성: %s", temp_dir)

        # 소스 비디오 처리
        if source_type == "youtube":
            source_url = request.form.get("source_url")
            logger.info("YouTube 다운로드 시작 (소스): %s", source_url)
            download_youtube(source_url, source_path)
        else:
            source_file = request.files["source_file"]
            logger.info("파일 업로드 처리 (소스): %s", source_file.filename)
            source_file.save(source_path)

        # 타겟 비디오 처리
        if target_type == "youtube":
            target_url = request.form.get("target_url")
            logger.info("YouTube 다운로드 시작 (타겟): %s", target_url)
            download_youtube(target_url, target_path)
        else:
            target_file = request.files["target_file"]
            logger.info("파일 업로드 처리 (타겟): %s", target_file.filename)
            target_file.save(target_path)

        # 화질 향상 및 길이 조정
        logger.info("소스 비디오 화질 향상 중...")
        enhance_video(source_path, enhanced_source)

        logger.info("타겟 비디오 화질 향상 중...")
        enhance_video(target_path, enhanced_target)

        # 향상된 비디오로 얼굴 교체 처리
        logger.info("얼굴 교체 처리 시작...")
        process_video(enhanced_source, enhanced_target, output_path)

        if os.path.exists(output_path):
            return send_file(output_path, mimetype="video/mp4")
        else:
            raise ValueError("출력 파일이 생성되지 않았습니다.")

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 400

    finally:
        # 임시 파일 정리
        cleanup_temp_files(
            source_path,
            target_path,
            enhanced_source,
            enhanced_target,
            output_path,
            temp_dir,
        )


if __name__ == "__main__":
    app.run(debug=True)
