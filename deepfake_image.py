from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
import os


def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if isinstance(img, Image.Image):
            img = np.array(img)
        if img.shape[-1] == 3:  # RGB 이미지인 경우
            plt.imshow(img)
        else:  # 그레이스케일 이미지인 경우
            plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()


class FaceSwapper:
    def __init__(self):
        # dlib 모델 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.join(
            current_dir,
            "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat",
        )

        if not os.path.exists(predictor_path):
            raise Exception("shape_predictor_68_face_landmarks.dat 파일이 필요합니다.")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_face_mask(self, img, landmarks):
        mask = np.zeros_like(img[:, :, 0])

        # 얼굴 윤곽선 포인트들
        jaw = landmarks[0:17]
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]

        # 얼굴 윤곽 생성 (눈썹 포함)
        face_contour = np.vstack([jaw, right_eyebrow[::-1], left_eyebrow[::-1]])

        # 마스크 채우기
        cv2.fillConvexPoly(mask, face_contour, 255)

        # 마스크 경계 부드럽게
        mask = cv2.GaussianBlur(mask, (15, 15), 7)

        return mask

    def enhance_eyes(self, source, target, landmarks, mask):
        # 눈 영역 추출
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # 눈 영역 마스크 생성
        eye_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(eye_mask, left_eye, 255)
        cv2.fillConvexPoly(eye_mask, right_eye, 255)
        eye_mask = cv2.GaussianBlur(eye_mask, (5, 5), 2)

        # 눈 영역 히스그램 매칭
        for i in range(3):  # BGR 채널별로 처리
            source_eye = source[:, :, i][eye_mask > 127]
            target_eye = target[:, :, i][eye_mask > 127]

            if len(source_eye) > 0 and len(target_eye) > 0:
                # 히스토그램 계산
                src_hist, _ = np.histogram(source_eye, 256, [0, 256])
                tgt_hist, _ = np.histogram(target_eye, 256, [0, 256])

                # 누적 분포 함수 계산
                src_cdf = src_hist.cumsum()
                tgt_cdf = tgt_hist.cumsum()

                # 정규화
                src_cdf_normalized = src_cdf / src_cdf[-1]
                tgt_cdf_normalized = tgt_cdf / tgt_cdf[-1]

                # 룩업 테이블 생성
                lut = np.zeros(256, dtype=np.uint8)
                j = 0
                for idx in range(256):
                    while j < 255 and src_cdf_normalized[j] <= tgt_cdf_normalized[idx]:
                        j += 1
                    lut[idx] = min(j, 255)  # 255를 초과하지 않도록 제한

                # 매핑 적용
                source[:, :, i] = cv2.LUT(source[:, :, i], lut)

        return source

    def warp_image(self, source, source_points, target_points, target_shape):
        # Delaunay 삼각분할을 위한 경계 상자 포인트 추가
        rect = cv2.boundingRect(target_points)
        (x, y, w, h) = rect

        # 삼각분할을 위한 포인트 준비
        points = np.array(target_points, np.int32)
        hull = cv2.convexHull(points)

        # 출력 이미지 초기화
        warped = np.zeros(target_shape, dtype=source.dtype)

        # Delaunay 삼각분할
        subdiv = cv2.Subdiv2D((0, 0, target_shape[1], target_shape[0]))
        for point in target_points:
            subdiv.insert((int(point[0]), int(point[1])))

        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        # 각 삼각형에 대해 워핑 수행
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # 타겟 이미지의 삼각형 좌표 찾기
            idx1 = self.find_point_index(target_points, pt1)
            idx2 = self.find_point_index(target_points, pt2)
            idx3 = self.find_point_index(target_points, pt3)

            if idx1 is not None and idx2 is not None and idx3 is not None:
                t1 = [source_points[idx1], source_points[idx2], source_points[idx3]]
                t2 = [target_points[idx1], target_points[idx2], target_points[idx3]]

                # 삼각형 워핑
                warped = self.warp_triangle(
                    source, warped, np.float32(t1), np.float32(t2)
                )

        return warped

    def find_point_index(self, points, point):
        # 포인트 인덱스 찾기
        for i, p in enumerate(points):
            if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
                return i
        return None

    def warp_triangle(self, img1, img2, t1, t2):
        # 삼각형 영역 찾기
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # 오프셋 포인트
        t1_offset = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
        t2_offset = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

        # 마스크 생성
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0))

        # 워핑
        img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
        size = (r2[2], r2[3])

        if size[0] > 0 and size[1] > 0:
            matrix = cv2.getAffineTransform(
                np.float32(t1_offset), np.float32(t2_offset)
            )
            warped = cv2.warpAffine(img1_rect, matrix, size, flags=cv2.INTER_LINEAR)
            warped = warped * mask

            # 결과 이미지에 합성
            img2_rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
            img2_rect_masked = img2_rect * (1 - mask)
            img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
                img2_rect_masked + warped
            )

        return img2

    def adjust_colors(self, source, target, mask):
        # 이미지 타입 통일
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # LAB 색상 공간으로 변환
        source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )

        # 마스크 영역 처리
        mask_bool = mask > 127
        if not np.any(mask_bool):
            return source.astype(np.uint8)

        # 각 채널별 색상 매칭
        for i in range(3):
            src_mean = np.mean(source_lab[mask_bool, i])
            src_std = np.std(source_lab[mask_bool, i])
            tgt_mean = np.mean(target_lab[mask_bool, i])
            tgt_std = np.std(target_lab[mask_bool, i])

            if src_std > 0:
                source_lab[mask_bool, i] = (
                    (source_lab[mask_bool, i] - src_mean) * (tgt_std / src_std)
                ) + tgt_mean

        # LAB to BGR
        source_lab = np.clip(source_lab, 0, 255)
        adjusted = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 색상 강도 복원
        color_intensity = 1.2
        adjusted = cv2.addWeighted(
            adjusted.astype(np.float32),
            color_intensity,
            adjusted.astype(np.float32),
            0,
            0,
        )
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        # 포아송 블렌딩
        mask_indices = np.where(mask_bool)
        if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
            center = (int(np.mean(mask_indices[1])), int(np.mean(mask_indices[0])))
            try:
                output = cv2.seamlessClone(
                    adjusted,
                    target.astype(np.uint8),
                    mask.astype(np.uint8),
                    center,
                    cv2.NORMAL_CLONE,
                )
            except:
                output = adjusted
        else:
            output = adjusted

        return output

    def process_images(self, image1, image2):
        try:
            # PIL Image를 OpenCV 형식으로 변환
            image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
            image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

            # 이미지 크기 통일
            target_size = (800, 800)
            image1 = cv2.resize(image1, target_size)
            image2 = cv2.resize(image2, target_size)

            # 그레이스케일 변환
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # 얼굴 검출
            rects1 = self.detector(gray1, 1)
            rects2 = self.detector(gray2, 1)

            if len(rects1) == 0 or len(rects2) == 0:
                raise Exception("얼굴을 찾을 수 없습니다.")

            # 랜드마크 추출
            landmarks1 = self.predictor(gray1, rects1[0])
            landmarks2 = self.predictor(gray2, rects2[0])

            landmarks1_np = np.array([[p.x, p.y] for p in landmarks1.parts()])
            landmarks2_np = np.array([[p.x, p.y] for p in landmarks2.parts()])

            # 1. 마스크 생성 및 표시
            mask1 = self.get_face_mask(image1, landmarks1_np)
            mask2 = self.get_face_mask(image2, landmarks2_np)

            # 마스크 시각화
            show_images(
                [image1, mask1, image2, mask2],
                ["Source Face", "Source Mask", "Target Face", "Target Mask"],
            )

            # 2. 얼굴 워핑
            warped = self.warp_image(image1, landmarks1_np, landmarks2_np, image2.shape)
            show_images([warped], ["Warped Face"])

            # 3. 얼굴 합성
            mask2_3d = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR) / 255.0
            blended = (warped * mask2_3d + image2 * (1 - mask2_3d)).astype(np.uint8)
            show_images([blended], ["Blended Face"])

            # 4. 색상 보정
            final_result = self.color_correct(blended, image2, mask2)
            show_images([final_result], ["Color Corrected"])

            # BGR에서 RGB로 변환
            final_result = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

            return final_result

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            return None

    def color_correct(self, source, target, mask):
        try:
            # 마스크 영역
            mask_bool = mask > 127

            # BGR 각 채널별로 히스토그램 매칭
            result = source.copy()
            for i in range(3):
                source_hist = cv2.calcHist(
                    [source], [i], mask.astype(np.uint8), [256], [0, 256]
                )
                target_hist = cv2.calcHist(
                    [target], [i], mask.astype(np.uint8), [256], [0, 256]
                )

                # 누적 분포 함수
                source_cdf = source_hist.cumsum()
                target_cdf = target_hist.cumsum()

                # 정규화
                source_cdf_normalized = source_cdf / source_cdf[-1]
                target_cdf_normalized = target_cdf / target_cdf[-1]

                # 룩업 테이블 생성
                lookup_table = np.zeros(256, dtype=np.uint8)
                j = 0
                for idx in range(256):
                    while (
                        j < 255
                        and source_cdf_normalized[j] <= target_cdf_normalized[idx]
                    ):
                        j += 1
                    lookup_table[idx] = j

                # 마스크 영역에만 적용
                result[mask_bool, i] = cv2.LUT(source[mask_bool, i], lookup_table)

            # 색상 강도 복원
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            enhanced = cv2.addWeighted(result, 1.5, result, 0, 0)  # 색상 강도 1.5배
            result = (enhanced * mask_3d + source * (1 - mask_3d)).astype(np.uint8)

            return result

        except Exception as e:
            print(f"색상 보정 중 오류 발생: {str(e)}")
            return source

    def final_blend(self, source, target, mask):
        # 포아송 블렌딩을 위한 중심점 계산
        mask_bool = mask > 127
        center = (
            int(np.mean(np.where(mask_bool)[1])),
            int(np.mean(np.where(mask_bool)[0])),
        )

        # 포아송 블렌딩
        try:
            result = cv2.seamlessClone(
                source, target, mask.astype(np.uint8), center, cv2.MIXED_CLONE
            )
        except:
            # 포아송 블렌딩 실패시 알파 블렌딩
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = (source * mask_3d + target * (1 - mask_3d)).astype(np.uint8)

        return result


if __name__ == "__main__":
    # 이미지 경로 설정
    image1_path = "C:\\Users\\heoch\\OneDrive\\datacenter\\sullyoon.jpg"
    image2_path = "C:\\Users\\heoch\\OneDrive\\datacenter\\minji.jpg"

    # 얼굴 스왑 실행
    swapper = FaceSwapper()
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    result = swapper.process_images(img1, img2)

    if result is not None:
        cv2.imwrite("face_swap_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print("얼굴 합성이 완료되었습니다.")
    else:
        print("얼굴 합성에 실패했습니다.")
