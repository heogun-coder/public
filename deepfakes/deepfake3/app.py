from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import dlib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max-limit

# 업로드 폴더가 없다면 생성
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# dlib의 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
    }


def face_blend(src_path, dst_path):
    # 이미지 읽기
    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)

    # 이미지 크기 맞추기
    dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]))

    # 얼굴 검출
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    src_faces = detector(src_gray)
    dst_faces = detector(dst_gray)

    if len(src_faces) == 0 or len(dst_faces) == 0:
        return None

    # 첫 번째 얼굴에 대해 랜드마크 검출
    src_landmarks = predictor(src_gray, src_faces[0])
    dst_landmarks = predictor(dst_gray, dst_faces[0])

    # 랜드마크 포인트 추출
    src_points = np.array([[p.x, p.y] for p in src_landmarks.parts()])
    dst_points = np.array([[p.x, p.y] for p in dst_landmarks.parts()])

    # 들로네 삼각분할을 위한 점들
    points = np.array(dst_points, np.int32)
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)

    # 포인트 추가
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    # 삼각형 리스트 얻기
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # 결과 이미지 준비
    result_img = np.zeros_like(dst_img)

    # 각 삼각형에 대해 워핑 수행
    for tri in triangles:
        x1, y1, x2, y2, x3, y3 = tri
        pt1 = (x1, y1)
        pt2 = (x2, y2)
        pt3 = (x3, y3)

        # 삼각형 점들의 인덱스 찾기
        src_pts = []
        dst_pts = []

        for i in range(len(dst_points)):
            if (
                (abs(dst_points[i][0] - x1) < 1 and abs(dst_points[i][1] - y1) < 1)
                or (abs(dst_points[i][0] - x2) < 1 and abs(dst_points[i][1] - y2) < 1)
                or (abs(dst_points[i][0] - x3) < 1 and abs(dst_points[i][1] - y3) < 1)
            ):
                src_pts.append(src_points[i])
                dst_pts.append(dst_points[i])

        if len(src_pts) == 3 and len(dst_pts) == 3:
            # 삼각형 영역 계산
            src_rect = cv2.boundingRect(np.float32(src_pts))
            dst_rect = cv2.boundingRect(np.float32(dst_pts))

            # 삼각형 마스크 생성
            mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
            dst_pts_shifted = [
                (p[0] - dst_rect[0], p[1] - dst_rect[1]) for p in dst_pts
            ]
            cv2.fillConvexPoly(mask, np.int32(dst_pts_shifted), 255)

            # 소스 이미지의 삼각형 부분 추출
            src_pts_shifted = [
                (p[0] - src_rect[0], p[1] - src_rect[1]) for p in src_pts
            ]
            src_tri_cropped = src_img[
                src_rect[1] : src_rect[1] + src_rect[3],
                src_rect[0] : src_rect[0] + src_rect[2],
            ]

            # 타겟 크기로 워핑
            size = (dst_rect[2], dst_rect[3])
            warp_mat = cv2.getAffineTransform(
                np.float32(src_pts_shifted), np.float32(dst_pts_shifted)
            )
            warped_tri = cv2.warpAffine(
                src_tri_cropped, warp_mat, size, flags=cv2.INTER_LINEAR
            )
            warped_tri = cv2.bitwise_and(warped_tri, warped_tri, mask=mask)

            # 색상 조정
            dst_tri_cropped = dst_img[
                dst_rect[1] : dst_rect[1] + dst_rect[3],
                dst_rect[0] : dst_rect[0] + dst_rect[2],
            ]

            # 평균 색상 계산 및 조정
            src_mean = cv2.mean(warped_tri, mask=mask)[:3]
            dst_mean = cv2.mean(dst_tri_cropped, mask=mask)[:3]
            color_correction = np.array(dst_mean) / (np.array(src_mean) + 1e-6)
            warped_tri = warped_tri.astype(np.float32)
            for i in range(3):
                warped_tri[:, :, i] *= color_correction[i]
            warped_tri = np.clip(warped_tri, 0, 255).astype(np.uint8)

            # 결과 이미지에 블렌딩
            img_rect = result_img[
                dst_rect[1] : dst_rect[1] + dst_rect[3],
                dst_rect[0] : dst_rect[0] + dst_rect[2],
            ]
            img_rect_gray = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
            mask = mask.astype(float) / 255.0
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            warped_tri = warped_tri.astype(float)
            img_rect = img_rect.astype(float)
            img_rect = img_rect * (1.0 - mask) + warped_tri * mask
            result_img[
                dst_rect[1] : dst_rect[1] + dst_rect[3],
                dst_rect[0] : dst_rect[0] + dst_rect[2],
            ] = img_rect.astype(np.uint8)

    # 얼굴 영역 마스크 생성 및 페더링
    face_mask = np.zeros_like(src_gray)
    face_hull = cv2.convexHull(dst_points)
    cv2.fillConvexPoly(face_mask, face_hull, 255)
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 10)
    face_mask = face_mask.astype(float) / 255.0

    # 최종 블렌딩
    face_mask_3d = np.repeat(face_mask[:, :, np.newaxis], 3, axis=2)
    result_img = result_img.astype(float)
    dst_img = dst_img.astype(float)
    final_result = dst_img * (1.0 - face_mask_3d) + result_img * face_mask_3d
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)

    # 결과 이미지 저장
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(result_path, final_result)
    return "result.jpg"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    if "source" not in request.files or "target" not in request.files:
        return jsonify({"error": "모든 이미지를 업로드해주세요"}), 400

    source = request.files["source"]
    target = request.files["target"]

    if source.filename == "" or target.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다"}), 400

    if not (allowed_file(source.filename) and allowed_file(target.filename)):
        return jsonify({"error": "허용되지 않는 파일 형식입니다"}), 400

    source_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(source.filename)
    )
    target_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(target.filename)
    )

    source.save(source_path)
    target.save(target_path)

    result = face_blend(source_path, target_path)
    if result is None:
        return jsonify({"error": "얼굴을 찾을 수 없습니다"}), 400

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
