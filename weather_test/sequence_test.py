from api_test import WeatherDataCollector
from sequence_predictor import SequenceWeatherPredictor
from datetime import datetime, timedelta
import pandas as pd


def main():
    collector = WeatherDataCollector()
    predictor = SequenceWeatherPredictor()

    try:
        # 데이터 수집
        print("날씨 데이터 수집 중...")
        train_data, test_data = collector.get_historical_data()

        print(f"훈련 데이터: {len(train_data)}시간")
        print(f"테스트 데이터: {len(test_data)}시간")

        # 모델 훈련
        print("\n시퀀스 기반 모델 훈련 시작...")
        predictor.train(train_data)

        # 테스트 데이터로 예측 수행
        print("\n테스트 데이터로 예측 수행 중...")
        weather_types = {0: "맑음", 1: "비", 2: "눈", 3: "흐림"}

        # 테스트 데이터를 16 스텝(3시간 * 16 = 48시간) 단위로 분할하여 예측
        test_sequences = []
        actual_weather = []

        # 테스트 데이터가 충분한지 확인
        if len(test_data) < predictor.sequence_steps + predictor.prediction_steps:
            print(
                f"테스트 데이터가 부족합니다. 필요: {predictor.sequence_steps + predictor.prediction_steps}, 현재: {len(test_data)}"
            )
            # 부족한 경우 훈련 데이터의 마지막 부분을 사용
            additional_data = train_data.iloc[
                -(predictor.sequence_steps + predictor.prediction_steps) :
            ]
            test_data = pd.concat([additional_data, test_data])
            print(
                f"훈련 데이터에서 추가 데이터를 가져와 테스트합니다. 테스트 데이터 크기: {len(test_data)}"
            )

        # 하나의 시퀀스만 생성
        sequence = test_data.iloc[: predictor.sequence_steps]
        if len(sequence) == predictor.sequence_steps:
            test_sequences.append(sequence)
            # 다음 8 스텝(24시간)의 실제 날씨 (최빈값)
            next_weather = test_data.iloc[
                predictor.sequence_steps : predictor.sequence_steps
                + predictor.prediction_steps
            ]["weather"].mode()[0]
            actual_weather.append(next_weather)

        # 예측 결과 출력
        correct = 0
        total = len(test_sequences)

        print("\n예측 결과:")
        for i, (sequence, actual) in enumerate(zip(test_sequences, actual_weather)):
            prediction = predictor.predict(sequence)
            predicted_weather = weather_types[prediction]
            actual_weather_code = predictor.prepare_labels([actual])[0]
            actual_weather_type = weather_types[actual_weather_code]

            print(f"\n시퀀스 {i+1}:")
            start_time = sequence.iloc[0]["datetime"]
            end_time = sequence.iloc[-1]["datetime"]
            prediction_time = end_time + timedelta(hours=24)

            print(f"분석 기간: {start_time} ~ {end_time}")
            print(f"예측 시점: {prediction_time}")
            print(f"예측 날씨: {predicted_weather}")
            print(f"실제 날씨: {actual_weather_type}")

            if prediction == actual_weather_code:
                correct += 1

        if total > 0:
            accuracy = correct / total
            print(f"\n전체 테스트 정확도: {accuracy:.2f}")
        else:
            print("\n테스트를 위한 충분한 데이터가 없습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
        raise e


if __name__ == "__main__":
    main()
