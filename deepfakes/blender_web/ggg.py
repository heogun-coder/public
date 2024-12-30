import numpy as np
import cv2
import dlib
from PIL import Image
import mediapipe as mp
import os


class FaceBlender:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = dlib.get_frontal_face_detector()

        # dlib 모델 파일 경로 확인 및 설정
        model_path = (
            "C:/Users/heoch/OneDrive/datacenter/shape_predictor_68_face_landmarks.dat"
        )
        if not os.path.exists(model_path):
            raise Exception("shape_predictor_68_face_landmarks.dat 파일이 필요합니다.")
        self.predictor = dlib.shape_predictor(model_path)

    def get_face_mask(self, image):
        """얼굴 영역의 마스크를 생성하는 함수"""
        try:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
            ) as face_mesh:

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if not results.multi_face_landmarks:
                    raise Exception("얼굴을 찾을 수 없습니다.")

                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)

                face_landmarks = results.multi_face_landmarks[0]
                points = np.array(
                    [
                        (int(landmark.x * w), int(landmark.y * h))
                        for landmark in face_landmarks.landmark
                    ]
                )

                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 255)
                mask = cv2.GaussianBlur(mask, (29, 29), 0)

                return mask
        except Exception as e:
            print(f"마스크 생성 중 오류 발생: {str(e)}")
            return None

    def align_faces(self, img1, img2):
        """두 얼굴 이미지를 정렬하는 함수"""
        try:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            faces1 = self.detector(img1_gray)
            faces2 = self.detector(img2_gray)

            if len(faces1) == 0 or len(faces2) == 0:
                raise Exception("얼굴을 찾을 수 없습니다.")

            landmarks1 = self.predictor(img1_gray, faces1[0])
            landmarks2 = self.predictor(img2_gray, faces2[0])

            points1 = np.array([[p.x, p.y] for p in landmarks1.parts()])
            points2 = np.array([[p.x, p.y] for p in landmarks2.parts()])

            eyes1 = points1[36:48]
            eyes2 = points2[36:48]

            M = cv2.estimateAffinePartial2D(eyes2, eyes1)[0]
            aligned_img2 = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

            return aligned_img2
        except Exception as e:
            print(f"얼굴 정렬 중 오류 발생: {str(e)}")
            return None

    def blend_with_mask(self, img1, img2, mask):
        """마스크를 사용하여 두 이미지를 블렌딩하는 함수"""
        try:
            mask_normalized = mask.astype(float) / 255
            mask_normalized = np.expand_dims(mask_normalized, axis=2)
            mask_normalized = np.repeat(mask_normalized, 3, axis=2)

            blended = img1 * (1 - mask_normalized) + img2 * mask_normalized
            return blended.astype(np.uint8)
        except Exception as e:
            print(f"블렌딩 중 오류 발생: {str(e)}")
            return None

    def blend_images(self, image1, image2, alpha=0.5):
        """두 얼굴 이미지를 블렌딩하는 메인 함수"""
        try:
            # PIL Image를 numpy 배열로 변환
            img1 = np.array(image1)
            img2 = np.array(image2)

            # 이미지 크기 확인
            if img1.shape[2] != 3 or img2.shape[2] != 3:
                raise Exception("RGB 이미지가 필요합니다.")

            # BGR로 변환
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

            # 이미지 크기 통일
            size = (800, 800)
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)

            # 얼굴 정렬
            aligned_img2 = self.align_faces(img1, img2)
            if aligned_img2 is None:
                raise Exception("얼굴 정렬 실패")

            # 얼굴 마스크 생성
            mask1 = self.get_face_mask(img1)
            mask2 = self.get_face_mask(aligned_img2)

            if mask1 is None or mask2 is None:
                raise Exception("마스크 생성 실패")

            # 마스크 블렌딩
            combined_mask = cv2.addWeighted(mask1, 1 - alpha, mask2, alpha, 0)

            # 얼굴 특징 블렌딩
            face_blend = cv2.addWeighted(img1, 1 - alpha, aligned_img2, alpha, 0)

            # 최종 블렌딩
            result = self.blend_with_mask(img1, face_blend, combined_mask)
            if result is None:
                raise Exception("최종 블렌딩 실패")

            # 결과 이미지를 RGB로 변환하고 PIL Image로 반환
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)

        except Exception as e:
            print(f"이미지 블렌딩 중 오류 발생: {str(e)}")
            return None
