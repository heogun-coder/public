# application programming interface : API

import requests
import json
import csv
from datetime import datetime
import os


def get_weather(city):
    # API 키와 기본 URL 설정
    api_key = "9fc8793cb0c2aeae66c21d1a9a3473eb"
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    # API 요청에 필요한 매개변수 설정
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # 섭씨 온도 사용
        "lang": "kr",  # 한국어로 결과 받기
    }

    try:
        # API 요청 보내기
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 오류 발생시 예외 발생

        # JSON 응답을 파이썬 딕셔너리로 변환
        weather_data = response.json()

        # 필요한 정보 추출
        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]

        # 결과 출력
        print(f"\n{city}의 현재 날씨 정보:")
        print(f"기온: {temperature}°C")
        print(f"습도: {humidity}%")
        print(f"날씨 상태: {description}")

        # CSV 파일 저장 로직 추가
        csv_filename = os.path.join("git_test", "weather_data.csv")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CSV 파일이 없으면 헤더와 함께 새로 생성
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["날짜/시간", "도시", "기온(°C)", "습도(%)", "날씨상태"]
                )
            writer.writerow([current_time, city, temperature, humidity, description])

        print(f"\n날씨 정보가 {csv_filename}에 저장되었습니다.")

    except requests.exceptions.RequestException as e:
        print(f"에러 발생: {e}")
    except KeyError as e:
        print(f"날씨 정보를 찾을 수 없습니다. 도시 이름을 확인해주세요.")
    except IOError as e:
        print(f"파일 저장 중 오류 발생: {e}")


# 프로그램 실행
if __name__ == "__main__":
    city = input("도시 이름을 입력하세요 (예: Seoul): ")
    get_weather(city)
