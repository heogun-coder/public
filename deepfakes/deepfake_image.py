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
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:  # 그레이스케일 이미지인 경우
            plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()


class FaceReplacer:
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

    def detect_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        if len(rects) == 0:
            return None, gray

        landmarks = self.predictor(gray, rects[0])
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])
        return landmarks_np, gray

    def create_face_mask(self, image, landmarks):
        mask = np.zeros_like(image[:, :, 0])

        # 얼굴 윤곽선 포인트들
        jaw = landmarks[0:17]  # 턱선
        left_eyebrow = landmarks[17:22]  # 왼쪽 눈썹
        right_eyebrow = landmarks[22:27]  # 오른쪽 눈썹
        nose = landmarks[27:36]  # 코
        left_eye = landmarks[36:42]  # 왼쪽 눈
        right_eye = landmarks[42:48]  # 오른쪽 눈
        mouth = landmarks[48:60]  # 입

        # 얼굴 윤곽 - 이마 부분을 제외하고 턱선만 사용
        face_contour = np.vstack(
            [
                jaw,  # 턱선
                np.array([[jaw[-1][0], jaw[0][1] - 50]]),  # 오른쪽 턱 위 지점
                np.array([[jaw[0][0], jaw[0][1] - 50]]),  # 왼쪽 턱 위 지점
            ]
        )

        # 기본 마스크 생성
        cv2.fillConvexPoly(mask, face_contour, 255)

        # 눈, 눈썹, 코, 입 영역 생성
        features_mask = np.zeros_like(mask)

        # 눈썹 영역
        cv2.fillConvexPoly(features_mask, left_eyebrow, 255)
        cv2.fillConvexPoly(features_mask, right_eyebrow, 255)

        # 눈 영역
        cv2.fillConvexPoly(features_mask, left_eye, 255)
        cv2.fillConvexPoly(features_mask, right_eye, 255)

        # 코 영역 (콧구멍 포함)
        nose_points = np.vstack([nose[0:1], nose[3:9]])  # 콧대와 콧구멍 부분
        cv2.fillConvexPoly(features_mask, nose_points, 255)

        # 입 영역
        cv2.fillConvexPoly(features_mask, mouth, 255)

        # 특징 마스크 블러 처리
        features_mask = cv2.GaussianBlur(features_mask, (5, 5), 2)

        # 최종 마스크 생성
        mask = cv2.subtract(mask, features_mask)

        # 경계 부드럽게 처리
        mask = cv2.GaussianBlur(mask, (9, 9), 3)

        return mask

    def extract_face(self, image, mask):
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        extracted = (image * mask_3d).astype(np.uint8)
        return extracted

    def analyze_face_angle(self, landmarks):
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        return angle

    def analyze_face_color(self, image, landmarks):
        mask = self.create_face_mask(image, landmarks)
        mask_bool = mask > 127
        face_color = cv2.mean(image, mask=mask.astype(np.uint8))[:3]
        return face_color

    def warp_face(self, face, source_landmarks, target_landmarks, target_shape):
        # Delaunay 삼각분할을 위한 포인트 준비
        points = np.array(target_landmarks, np.int32)
        rect = cv2.boundingRect(points)
        subdiv = cv2.Subdiv2D((0, 0, target_shape[1], target_shape[0]))

        for point in target_landmarks:
            subdiv.insert((int(point[0]), int(point[1])))

        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        # 워핑된 이미지 초기화
        warped = np.zeros(target_shape, dtype=np.uint8)

        # 각 삼각형에 대해 워핑 수행
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # 삼각형 인덱스 찾기
            idx1 = self.find_point_index(target_landmarks, pt1)
            idx2 = self.find_point_index(target_landmarks, pt2)
            idx3 = self.find_point_index(target_landmarks, pt3)

            if all(x is not None for x in [idx1, idx2, idx3]):
                triangle1 = np.float32(
                    [
                        source_landmarks[idx1],
                        source_landmarks[idx2],
                        source_landmarks[idx3],
                    ]
                )
                triangle2 = np.float32(
                    [
                        target_landmarks[idx1],
                        target_landmarks[idx2],
                        target_landmarks[idx3],
                    ]
                )

                # 삼각형 워핑
                warped = self.warp_triangle(face, warped, triangle1, triangle2)

        return warped

    def find_point_index(self, points, point):
        for i, p in enumerate(points):
            if abs(p[0] - point[0]) < 1 and abs(p[1] - point[1]) < 1:
                return i
        return None

    def warp_triangle(self, img1, img2, t1, t2):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        t1_offset = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
        t2_offset = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0))

        img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
        size = (r2[2], r2[3])

        if size[0] > 0 and size[1] > 0:
            matrix = cv2.getAffineTransform(
                np.float32(t1_offset), np.float32(t2_offset)
            )
            warped = cv2.warpAffine(img1_rect, matrix, size, flags=cv2.INTER_LINEAR)
            warped = warped * mask

            img2_rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
            img2_rect_masked = img2_rect * (1 - mask)
            img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
                img2_rect_masked + warped
            )

        return img2

    def blend_and_correct(self, warped_face, target_image, source_mask, target_mask):
        try:
            # 원본 이미지 복사본 생성 (색상 복원용)
            image3 = target_image.copy()

            # 1. 타겟 마스크(image2의 마스크)를 기준으로 블렌딩
            target_mask_3d = (
                cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            )
            target_mask_3d = cv2.GaussianBlur(
                target_mask_3d, (5, 5), 2
            )  # 경계 부드럽게

            # 2. 초기 블렌딩
            initial_blend = (
                warped_face * target_mask_3d + target_image * (1 - target_mask_3d)
            ).astype(np.uint8)

            # 3. 포아송 블렌딩
            mask_bool = target_mask > 127
            center = (
                int(np.mean(np.where(mask_bool)[1])),
                int(np.mean(np.where(mask_bool)[0])),
            )

            try:
                blended = cv2.seamlessClone(
                    initial_blend,
                    target_image,
                    target_mask.astype(np.uint8),
                    center,
                    cv2.MIXED_CLONE,
                )
            except:
                blended = initial_blend

            # 4. image3의 색상으로 복원
            blended_lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB).astype(np.float32)
            original_lab = cv2.cvtColor(image3, cv2.COLOR_BGR2LAB).astype(np.float32)

            # 마스크 영역의 색상 통계
            for i in range(3):
                if np.any(mask_bool):
                    orig_mean = np.mean(original_lab[mask_bool, i])
                    orig_std = np.std(original_lab[mask_bool, i])
                    blend_mean = np.mean(blended_lab[mask_bool, i])
                    blend_std = np.std(blended_lab[mask_bool, i])

                    if blend_std > 0:
                        # image3의 색상 특성으로 복원
                        blended_lab[mask_bool, i] = (
                            blended_lab[mask_bool, i] - blend_mean
                        ) * (orig_std / blend_std) + orig_mean

            # LAB to BGR
            blended_lab = np.clip(blended_lab, 0, 255)
            color_corrected = cv2.cvtColor(
                blended_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
            )

            # 5. 최종 블렌딩 (image3의 색상 유지)
            final_mask = cv2.GaussianBlur(target_mask_3d, (7, 7), 3)
            result = (color_corrected * final_mask + image3 * (1 - final_mask)).astype(
                np.uint8
            )

            # image3의 색상 특성 강화
            result = cv2.addWeighted(result, 0.85, image3, 0.15, 0)

            return result

        except Exception as e:
            print(f"블렌딩 중 오류 발생: {str(e)}")
            return target_image

    def process_images(self, image1, image2):
        try:
            # 이미지 준비
            image1 = np.array(image1)
            image2 = np.array(image2)
            image3 = image2.copy()  # 원본 색상 참조용

            # 이미지 크기 통일
            target_size = (800, 800)
            image1 = cv2.resize(image1, target_size)
            image2 = cv2.resize(image2, target_size)
            image3 = cv2.resize(image3, target_size)

            # 1. 얼굴 랜드마크 검출
            landmarks1, gray1 = self.detect_landmarks(image1)
            landmarks2, gray2 = self.detect_landmarks(image2)

            if landmarks1 is None or landmarks2 is None:
                raise Exception("얼굴을 찾을 수 없습니다.")

            # 2. 각도 및 크기 보정
            angle1 = self.calculate_face_angle(landmarks1)
            angle2 = self.calculate_face_angle(landmarks2)
            angle_diff = angle2 - angle1

            # 이미지 중심점 계산
            h, w = image1.shape[:2]
            center = (w // 2, h // 2)

            # 회전 변환
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_diff, 1.0)
            aligned_image1 = cv2.warpAffine(image1, rotation_matrix, (w, h))

            # 회전된 이미지에서 다시 랜드마크 검출
            landmarks1, _ = self.detect_landmarks(aligned_image1)

            # 3. 얼굴 워핑 (image2에 맞춰 변형)
            warped_face = self.warp_face(
                aligned_image1, landmarks1, landmarks2, image2.shape
            )

            # 4. 마스크 생성
            target_mask = self.create_face_mask(image2, landmarks2)
            features_mask = self.create_features_mask(landmarks2)

            # 마스크 블러 처리
            mask_3d = (
                cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
            )
            mask_3d = cv2.GaussianBlur(mask_3d, (9, 9), 3)

            features_mask_3d = (
                cv2.cvtColor(features_mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                / 255.0
            )
            features_mask_3d = cv2.GaussianBlur(features_mask_3d, (5, 5), 2)

            # 5. 조명 보정
            mask_bool = target_mask > 127

            # LAB 색상 공간에서 조명 보정
            warped_lab = cv2.cvtColor(warped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(image3, cv2.COLOR_BGR2LAB).astype(np.float32)

            # 마스크 영역의 조명 보정
            for i in range(3):
                if np.any(mask_bool):
                    if i == 0:  # L 채널만 조명 보정
                        target_mean = np.mean(target_lab[mask_bool, 0])
                        warped_mean = np.mean(warped_lab[mask_bool, 0])
                        warped_lab[mask_bool, 0] += target_mean - warped_mean
                    else:  # a,b 채널은 색상 보정
                        target_mean = np.mean(target_lab[mask_bool, i])
                        target_std = np.std(target_lab[mask_bool, i])
                        warped_mean = np.mean(warped_lab[mask_bool, i])
                        warped_std = np.std(warped_lab[mask_bool, i])

                        if warped_std > 0:
                            warped_lab[mask_bool, i] = (
                                warped_lab[mask_bool, i] - warped_mean
                            ) * (target_std / warped_std) + target_mean

            # LAB to BGR
            warped_lab = np.clip(warped_lab, 0, 255)
            light_corrected = cv2.cvtColor(
                warped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
            )

            # 6. 블렌딩
            # 첫 번째 블렌딩: 조명 보정된 얼굴과 타겟 이미지
            initial_blend = (light_corrected * mask_3d + image3 * (1 - mask_3d)).astype(
                np.uint8
            )

            # 두 번째 블렌딩: 특징 복원
            result = (
                initial_blend * (1 - features_mask_3d) + image3 * features_mask_3d
            ).astype(np.uint8)

            # 결과 시각화
            show_images(
                [
                    image2,  # 타겟 이미지
                    warped_face,  # 워핑된 얼굴
                    light_corrected,  # 조명 보정된 얼굴
                    result,  # 최종 결과
                ],
                ["Target Image", "Warped Face", "Light Corrected", "Final Result"],
            )

            return result

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            return None

    def calculate_face_angle(self, landmarks):
        # 눈의 중심점을 이용한 각도 계산
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)

        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def create_features_mask(self, landmarks):
        # 눈, 코, 입 특징 마스크 생성
        features_mask = np.zeros((800, 800), dtype=np.uint8)

        # 눈 영역
        cv2.fillConvexPoly(features_mask, landmarks[36:42], 255)  # 왼쪽 눈
        cv2.fillConvexPoly(features_mask, landmarks[42:48], 255)  # 오른쪽 눈

        # 콧구멍 영역
        nose_points = landmarks[31:36]
        cv2.fillConvexPoly(features_mask, nose_points, 255)

        # 입 영역
        mouth_points = landmarks[48:60]
        cv2.fillConvexPoly(features_mask, mouth_points, 255)

        return features_mask


if __name__ == "__main__":
    # 이미지 경로 설정
    image1_path = "C:\\Users\\heoch\\OneDrive\\datacenter\\hanni.jpg"
    image2_path = "C:\\Users\\heoch\\OneDrive\\datacenter\\haerin.jpg"

    # 이미지 읽기 (BGR -> RGB 변환 없이)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 얼굴 교체 실행
    replacer = FaceReplacer()
    result = replacer.process_images(img1, img2)

    if result is not None:
        cv2.imwrite("face_replacement_result.jpg", result)
        print("얼굴 교체가 완료되었습니다.")
    else:
        print("얼굴 교체에 실패했습니다.")
