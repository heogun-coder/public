from gpiozero import DistanceSensor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time

# 초음파 센서 설정 (GPIO 핀 번호는 실제 연결에 맞게 수정하세요)
sensor = DistanceSensor(echo=24, trigger=23)

# 데이터 저장을 위한 리스트
distances = []
timestamps = []
window_size = 50  # 그래프에 표시할 데이터 포인트 수

# 그래프 초기 설정
fig, ax = plt.subplots()
(line,) = ax.plot([], [])
ax.set_xlim(0, window_size)
ax.set_ylim(0, 2)  # 거리 범위에 맞게 조정하세요
ax.set_title("실시간 거리 측정")
ax.set_xlabel("시간")
ax.set_ylabel("거리 (m)")


def analyze_pattern(data, window=10):
    """패턴을 분석하는 함수"""
    if len(data) < window:
        return "데이터 수집 중..."

    # 최근 데이터의 기울기 계산
    recent_data = data[-window:]
    slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]

    # 표준편차 계산
    std_dev = np.std(recent_data)

    # 패턴 판별
    if std_dev > 0.5:  # 임계값은 실제 데이터에 맞게 조정하세요
        return "돌발(spike)"
    elif abs(slope) > 0.1:  # 임계값은 실제 데이터에 맞게 조정하세요
        return "접근(ascending)" if slope > 0 else "하강(descending)"
    else:
        return "평소(just)"


voltages = []


def update(frame):
    voltage = sensor.distance

    voltages.append(voltage)
    timestamps.append(frame)

    # 윈도우 크기 유지
    if len(voltages) > window_size:
        voltages.pop(0)
        timestamps.pop(0)

    # 그래프 업데이트
    line.set_data(range(len(voltages)), voltages)

    # 패턴 분석
    pattern = analyze_pattern(distances)
    ax.set_title(f"실시간 거리 측정 - 현재 패턴: {pattern}")

    return (line,)


# 애니메이션 시작
ani = FuncAnimation(fig, update, frames=None, interval=100, blit=True)
plt.show()
