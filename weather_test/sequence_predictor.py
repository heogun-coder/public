import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime, timedelta


class SequenceWeatherPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        # 3시간 단위로 변경
        self.sequence_steps = 16  # 2일 = 48시간 = 16개의 3시간 단위
        self.prediction_steps = 8  # 1일 = 24시간 = 8개의 3시간 단위

    def prepare_sequence_data(self, data):
        """48시간(16개의 3시간 단위)치 데이터를 하나의 특성 벡터로 만듭니다."""
        sequences = []
        labels = []

        # 시간순으로 정렬
        data = data.sort_values("datetime")

        print(f"데이터 시작 시간: {data['datetime'].min()}")
        print(f"데이터 종료 시간: {data['datetime'].max()}")
        print(f"전체 데이터 포인트 수: {len(data)}")
        print(
            f"필요한 최소 데이터 포인트 수: {self.sequence_steps + self.prediction_steps}"
        )

        # 시퀀스 데이터 생성 (3시간 단위)
        for i in range(len(data) - self.sequence_steps - self.prediction_steps + 1):
            # 2일치 데이터 (16개 포인트)
            sequence = data.iloc[i : i + self.sequence_steps]
            # 다음 1일치 데이터 (8개 포인트)의 날씨 모드(최빈값)
            target_weather = data.iloc[
                i
                + self.sequence_steps : i
                + self.sequence_steps
                + self.prediction_steps
            ]["weather"].mode()[0]

            # 특성 벡터 생성
            feature_vector = []
            for _, step in sequence.iterrows():
                feature_vector.extend(
                    [
                        step["temperature"],
                        step["humidity"],
                        step["pressure"],
                        step["wind_speed"],
                    ]
                )

            sequences.append(feature_vector)
            labels.append(target_weather)

        if not sequences:
            raise ValueError(
                "시퀀스 데이터를 생성할 수 없습니다. "
                f"데이터가 충분하지 않습니다. (현재 데이터 수: {len(data)}, "
                f"필요한 최소 데이터 수: {self.sequence_steps + self.prediction_steps})"
            )

        return np.array(sequences), np.array(labels)

    def prepare_labels(self, labels):
        """날씨 레이블을 숫자로 변환합니다."""
        weather_mapping = {
            "Clear": 0,
            "Rain": 1,
            "Snow": 2,
            "Clouds": 3,
            "Mist": 3,
            "Fog": 3,
            "Haze": 3,
            "Drizzle": 1,
            "Thunderstorm": 1,
        }

        return np.array([weather_mapping.get(label, 3) for label in labels])

    def train(self, data):
        """모델을 훈련합니다."""
        print(f"전체 데이터 수: {len(data)}")

        # 시퀀스 데이터 준비
        X, y = self.prepare_sequence_data(data)
        y = self.prepare_labels(y)

        print(f"생성된 시퀀스 수: {len(X)}")

        if len(X) == 0:
            raise ValueError("훈련 데이터가 충분하지 않습니다.")

        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 모델 훈련
        self.model.fit(X_scaled, y)

        # 훈련 성능 평가
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"훈련 정확도: {accuracy:.2f}")
        print("\n분류 보고서:")
        print(classification_report(y, y_pred))

    def predict(self, sequence_data):
        """48시간치 데이터로 다음 24시간의 날씨를 예측합니다."""
        if len(sequence_data) != self.sequence_steps:
            raise ValueError(
                f"입력 데이터는 {self.sequence_steps}개의 3시간 단위 데이터여야 합니다."
            )

        # 특성 벡터 생성
        feature_vector = []
        for _, step in sequence_data.iterrows():
            feature_vector.extend(
                [
                    step["temperature"],
                    step["humidity"],
                    step["pressure"],
                    step["wind_speed"],
                ]
            )

        # 스케일링 및 예측
        X_scaled = self.scaler.transform([feature_vector])
        prediction = self.model.predict(X_scaled)
        return prediction[0]
