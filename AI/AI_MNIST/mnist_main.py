import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터 설정
num_epochs = 5
batch_size = 100
learning_rate = 0.001


# CNN 모델 정의
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train_model():
    # MNIST 데이터셋 ���운로드 및 전처리
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 인스턴스 생성
    model = ConvNet().to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
                )

    # 테스트
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"테스트 정확도: {100 * correct / total}%")

    # 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")
    return model


def preprocess_image(image_path):
    # 이미지를 흑백으로 변환하고 크기를 28x28로 조정
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))

    # 이미지를 텐서로 변환하고 정규화
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가


def predict_digit(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()


def main():
    while True:
        print("\n1. 모델 학습")
        print("2. 손글씨 숫자 예측")
        print("3. 종료")
        choice = input("선택하세요 (1-3): ")

        if choice == "1":
            print("\n모델 학습을 시작합니다...")
            model = train_model()
            print("모델 학습이 완료되었습니다.")

        elif choice == "2":
            try:
                # 저장된 모델 불러오기
                model = ConvNet().to(device)
                model.load_state_dict(torch.load("mnist_cnn.pth"))

                # 이미지 경로 입력
                image_path = input("\n손글씨 이미지 경로를 입력하세요: ")

                # 이미지 전처리 및 예측
                image_tensor = preprocess_image(image_path)
                predicted_digit = predict_digit(model, image_tensor)
                print(f"예측된 숫자: {predicted_digit}")

            except Exception as e:
                print(f"에러 발생: {str(e)}")

        elif choice == "3":
            print("\n프로그램을 종료합니다.")
            break

        else:
            print("\n잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
