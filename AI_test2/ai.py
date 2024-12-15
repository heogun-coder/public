import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


mnist = keras.datasets.mnist


# 1. MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 채널 차원 추가 (28x28x1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype("float32") / 255.0  # 정규화
x_test = x_test.astype("float32") / 255.0

# y_train = to_categorical(y_train, 10)  # 레이블을 One-Hot 인코딩
# y_test = to_categorical(y_test, 10)

# 3. CNN 모델 정의
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),  # 10개의 클래스 출력
    ]
)

# 4. 모델 컴파일
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. 모델 훈련
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 6. 모델 평가 (테스트 데이터에 대해)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
