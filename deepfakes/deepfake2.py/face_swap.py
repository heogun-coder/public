import cv2
import numpy as np
import dlib


def get_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    points = []
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        points.append((x, y))
    return np.array(points)


def get_delaunay_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((p[0], p[1]))

    triangles = subdiv.getTriangleList()
    return triangles


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # 삼각형 오프셋 조정
    t1_offset = [(p[0] - r1[0], p[1] - r1[1]) for p in t1]
    t2_offset = [(p[0] - r2[0], p[1] - r2[1]) for p in t2]

    # 변환 행렬 계산
    M = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

    # 워핑 수행
    warped = cv2.warpAffine(
        img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]], M, (r2[2], r2[3])
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0))

    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] *= 1.0 - mask
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] += warped * mask


def color_correct(source, target, landmarks):
    # 얼굴 영역 마스크 생성
    mask = np.zeros(target.shape, dtype=target.dtype)
    points = landmarks.astype(np.int32)
    cv2.fillConvexPoly(mask, points, (1, 1, 1))

    # 얼굴 영역의 평균 색상과 표준편차 계산
    source_mean = cv2.mean(source, mask=mask[:, :, 0])
    target_mean = cv2.mean(target, mask=mask[:, :, 0])
    source_std = cv2.meanStdDev(source, mask=mask[:, :, 0])[1]
    target_std = cv2.meanStdDev(target, mask=mask[:, :, 0])[1]

    # 색상 보정
    result = source.copy()
    for i in range(3):  # BGR 각 채널에 대해
        result[:, :, i] = (
            (source[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i])
        ) + target_mean[i]

    return result


def seamless_clone(source, target, mask, center):
    # Poisson Blending 적용
    output = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
    return output


def correct_illumination(source, target, mask):
    # Lab 색공간으로 변환
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # L 채널(밝기) 조정
    source_l = source_lab[:, :, 0]
    target_l = target_lab[:, :, 0]

    # 마스크 영역의 평균 밝기 계산
    source_mean = cv2.mean(source_l, mask=mask)[0]
    target_mean = cv2.mean(target_l, mask=mask)[0]

    # 밝기 보정
    diff = target_mean - source_mean
    source_lab[:, :, 0] = cv2.add(source_lab[:, :, 0], diff)

    # BGR로 다시 변환
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)


def process_videos(source_path, target_path, output_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    source_cap = cv2.VideoCapture(source_path)
    target_cap = cv2.VideoCapture(target_path)

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, 30.0, (int(target_cap.get(3)), int(target_cap.get(4)))
    )

    while True:
        ret1, source_frame = source_cap.read()
        ret2, target_frame = target_cap.read()

        if not ret1 or not ret2:
            break

        source_points = get_landmarks(source_frame, detector, predictor)
        target_points = get_landmarks(target_frame, detector, predictor)

        if source_points is None or target_points is None:
            continue

        # 얼굴 영역 마스크 생성
        mask = np.zeros(target_frame.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(target_points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        # 워핑된 얼굴에 대한 처리
        warped_face = source_frame.copy()
        # Delaunay 삼각형 생성
        rect = (0, 0, target_frame.shape[1], target_frame.shape[0])
        triangles = get_delaunay_triangles(rect, target_points)

        # 각 삼각형에 대해 워핑 수행
        for t in triangles:
            t = t.astype(np.int32)

            # 삼각형의 세 점
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # 워핑 수행
            warp_triangle(source_frame, target_frame, [pt1, pt2, pt3], [pt1, pt2, pt3])

        # 색상 보정
        warped_face = color_correct(warped_face, target_frame, hull)

        # 조명 보정
        warped_face = correct_illumination(warped_face, target_frame, mask)

        # 얼굴 중심점 계산
        center = (int(np.mean(target_points[:, 0])), int(np.mean(target_points[:, 1])))

        # Seamless cloning으로 자연스러운 합성
        result = seamless_clone(warped_face, target_frame, mask, center)

        out.write(result)

    source_cap.release()
    target_cap.release()
    out.release()
