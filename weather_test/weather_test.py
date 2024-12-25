from api_test import WeatherDataCollector
from AI_predictor import WeatherPredictor
import pandas as pd
import numpy as np


def main():
    # 데이터 수집기 초기화
    collector = WeatherDataCollector()

    try:
        # 현재 날씨 데이터 수집
        print("현재 날씨 데이터 수집 중...")
        current_weather = collector.get_current_weather()
        print(f"현재 서울 날씨: {current_weather['weather'][0]['description']}")
        print(f"기온: {current_weather['main']['temp']}°C")

        # 과거 데이터 수집 및 분리
        print("\n과거 날씨 데이터 수집 중...")
        train_data, test_data = collector.get_historical_data()

        print(f"훈련 데이터 크기: {len(train_data)} 샘플")
        print(f"테스트 데이터 크기: {len(test_data)} 샘플")

        # AI 모델 훈련
        print("\nAI 모델 훈련 시작...")
        predictor = WeatherPredictor()

        # 훈련 데이터로 모델 학습
        train_features = predictor.prepare_features(train_data)
        train_labels = predictor.prepare_labels(train_data)
        predictor.train(train_features, train_labels)

        # 테스트 데이터로 성능 평가
        print("\n테스트 데이터로 성능 평가 중...")
        test_features = predictor.prepare_features(test_data)
        test_labels = predictor.prepare_labels(test_data)

        predictions = predictor.predict(test_features)
        weather_types = {0: "맑음", 1: "비", 2: "눈", 3: "흐림"}

        # 예측 결과 분석
        correct = 0
        total = len(test_labels)

        print("\n예측 결과 샘플:")
        for i in range(min(5, total)):
            actual_weather = weather_types[test_labels.iloc[i]]
            predicted_weather = weather_types[predictions[i]]
            print(f"날짜: {test_data['datetime'].iloc[i]}")
            print(f"실제 날씨: {actual_weather}")
            print(f"예측 날씨: {predicted_weather}\n")
            if predictions[i] == test_labels.iloc[i]:
                correct += 1

        accuracy = correct / total
        print(f"\n전체 테스트 데이터 정확도: {accuracy:.2f}")

    except Exception as e:
        print(f"오류 발생: {e}")
        raise e


if __name__ == "__main__":
    main()
