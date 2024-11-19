import cv2
import dlib
import numpy as np
from scipy.spatial import Delaunay
import os


class FaceBlender:
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

    def warp_triangle(self, img1, img2, t1, t2):
        matrix = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
        warped_triangle = cv2.warpAffine(
            img1,
            matrix,
            (img2.shape[1], img2.shape[0]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped_triangle

    def blend_images(self, img1, img2, mask):
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(img2, img2, mask=mask_inv)
        img2_fg = cv2.bitwise_and(img1, img1, mask=mask)
        blended_image = cv2.add(img1_bg, img2_fg)
        return blended_image

    def adjust_brightness_contrast(self, img, alpha=1.0, beta=0):
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def process_images(self, image1, image2, alpha=0.5):
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

            # 평균 랜드마크 계산
            average_landmarks = (landmarks1_np + landmarks2_np) // 2

            # Delaunay 삼각분할
            triangles = Delaunay(average_landmarks).simplices

            # 출력 이미지 초기화
            output_image = np.zeros_like(image2)

            # 삼각형별 워핑 및 블렌딩
            for triangle in triangles:
                t1 = np.array(
                    [landmarks1_np[triangle[i]] for i in range(3)], dtype=np.int32
                )
                t2 = np.array(
                    [landmarks2_np[triangle[i]] for i in range(3)], dtype=np.int32
                )
                t_avg = np.array(
                    [average_landmarks[triangle[i]] for i in range(3)], dtype=np.int32
                )

                warped_triangle1 = self.warp_triangle(image1, output_image, t1, t_avg)
                warped_triangle2 = self.warp_triangle(image2, output_image, t2, t_avg)

                mask = np.zeros(
                    (output_image.shape[0], output_image.shape[1]), dtype=np.uint8
                )
                cv2.fillConvexPoly(mask, np.int32(t_avg), 255)

                triangle_area = cv2.bitwise_and(
                    warped_triangle1, warped_triangle1, mask=mask
                )
                blended_triangle = cv2.addWeighted(
                    triangle_area, alpha, warped_triangle2, 1 - alpha, 0
                )
                output_image = self.blend_images(blended_triangle, output_image, mask)

            # BGR에서 RGB로 변환
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            return output_image

        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            return None
