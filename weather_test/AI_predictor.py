import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def prepare_features(self, data):
        """특성 데이터를 전처리합니다."""
        features = data[["humidity", "pressure", "temperature", "wind_speed"]]
        # 결측치 처리
        features = features.fillna(features.mean())
        return features

    def prepare_labels(self, data):
        """날씨 레이블을 준비합니다."""
        # OpenWeatherMap의 날씨 상태를 우리의 분류로 매핑
        weather_mapping = {
            "Clear": 0,  # 맑음
            "Rain": 1,  # 비
            "Snow": 2,  # 눈
            "Clouds": 3,  # 흐림
            # 추가적인 날씨 상태들
            "Mist": 3,
            "Fog": 3,
            "Haze": 3,
            "Drizzle": 1,
            "Thunderstorm": 1,
        }

        # 매핑되지 않은 날씨 상태 확인
        unknown_weather = set(data["weather"].unique()) - set(weather_mapping.keys())
        if unknown_weather:
            print(f"Warning: 매핑되지 않은 날씨 상태가 있습니다: {unknown_weather}")

        return data["weather"].map(weather_mapping)

    def train(self, features, labels):
        """모델을 훈련합니다."""
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 모델 훈련
        self.model.fit(X_train_scaled, y_train)

        # 성능 평가
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"모델 정확도: {accuracy:.2f}")
        print("\n분류 보고서:")
        print(classification_report(y_test, y_pred))

    def predict(self, features):
        """새로운 데이터에 대해 날씨를 예측합니다."""
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        return predictions
