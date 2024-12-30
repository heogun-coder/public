import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np


class WeatherDataCollector:
    def __init__(self):
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.api_key = "9fc8793cb0c2aeae66c21d1a9a3473eb"
        self.city_id = "1835848"  # 서울

    def get_current_weather(self):
        """현재 날씨 데이터를 수집합니다."""
        url = f"{self.base_url}/weather"
        params = {
            "id": self.city_id,
            "appid": self.api_key,
            "units": "metric",  # 섭씨 온도
            "lang": "kr",
        }

        response = requests.get(url, params=params)
        return response.json()

    def get_historical_data(self):
        """5일치 예보 데이터를 수집하고 훈련/테스트 데이터로 분할합니다."""
        url = f"{self.base_url}/forecast"
        params = {
            "id": self.city_id,
            "appid": self.api_key,
            "units": "metric",
            "lang": "kr",
            "cnt": 40,
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"API 요청 실패: {response.status_code}, {response.text}")

        data = response.json()
        if "list" not in data:
            raise Exception(f"잘못된 API 응답: {data}")

        weather_data = []
        for item in data["list"]:
            weather_data.append(
                {
                    "datetime": datetime.fromtimestamp(item["dt"]),
                    "temperature": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "wind_speed": item["wind"]["speed"],
                    "weather": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"],
                }
            )

        df = pd.DataFrame(weather_data)
        df = df.sort_values("datetime")

        print("\n데이터 수집 정보:")
        print(f"시작 시간: {df['datetime'].min()}")
        print(f"종료 시간: {df['datetime'].max()}")
        print(f"전체 데이터 포인트 수: {len(df)}")

        # 데이터 분할
        train_size = int(len(df) * 0.75)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        print(f"\n훈련 데이터: {len(train_data)} 포인트")
        print(f"테스트 데이터: {len(test_data)} 포인트")

        if len(train_data) < 24:  # 최소 24개 = 16+8
            raise Exception(
                "훈련 데이터가 부족합니다. "
                f"최소 24개의 데이터 포인트가 필요하지만 현재 {len(train_data)}개입니다."
            )

        return train_data, test_data

    def save_weather_data(self, data, filename):
        data.to_json(filename, orient="records", force_ascii=False)
        print(f"데이터가 {filename}에 저장되었습니다.")

    def generate_test_data(self):

        weather_conditions = {
            "Clear": {  # 맑음
                "temp_range": (15, 30),
                "humidity_range": (30, 60),
                "pressure_range": (1008, 1020),
                "wind_range": (0, 5),
            },
            "Rain": {  # 비
                "temp_range": (10, 25),
                "humidity_range": (60, 95),
                "pressure_range": (995, 1005),
                "wind_range": (3, 8),
            },
            "Snow": {  # 눈
                "temp_range": (-5, 2),
                "humidity_range": (70, 90),
                "pressure_range": (1000, 1015),
                "wind_range": (2, 6),
            },
            "Clouds": {  # 흐림
                "temp_range": (10, 28),
                "humidity_range": (50, 80),
                "pressure_range": (1000, 1015),
                "wind_range": (2, 7),
            },
        }

        test_data = []
        for weather, conditions in weather_conditions.items():
            for _ in range(50):
                data = {
                    "temperature": round(
                        np.random.uniform(*conditions["temp_range"]), 1
                    ),
                    "humidity": round(
                        np.random.uniform(*conditions["humidity_range"]), 1
                    ),
                    "pressure": round(
                        np.random.uniform(*conditions["pressure_range"]), 1
                    ),
                    "wind_speed": round(
                        np.random.uniform(*conditions["wind_range"]), 1
                    ),
                    "weather": weather,
                }
                test_data.append(data)

        return pd.DataFrame(test_data)
