import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score

# 경로 설정
input_dir = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/captured_faces"
model_save_path = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/feature_model.pth"
onnx_model_path = "/home/jinwoo/catkin_ws/src/project_ojakdong/src/feature_model.onnx"  # ONNX 모델 경로

# 데이터셋 준비 함수
def prepare_dataset(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Dataset directory '{input_dir}' does not exist.")
    if not os.listdir(input_dir):
        raise FileNotFoundError(f"Dataset directory '{input_dir}' is empty.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    
    dataset = datasets.ImageFolder(input_dir, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 정의
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceRecognitionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Convolution Layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling Layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # Fully Connected Layer
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output Layer
        )

    def forward(self, x):
        return self.model(x)

# 모델 학습
def train_model(data_loader, save_path, onnx_path):
    model = FaceRecognitionModel(num_classes=2)

    # GPU 또는 CPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 학습
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # F1 Score 계산
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, F1 Score: {f1:.4f}")

        # 학습률 스케줄러 업데이트
        scheduler.step()

    # PyTorch 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # ONNX 모델 저장
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 입력 텐서의 크기 정의
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, 
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    print(f"ONNX model saved to {onnx_path}")

# 메인 함수
if __name__ == "__main__":
    # 데이터 준비
    data_loader = prepare_dataset(input_dir)

    # 모델 학습 및 저장
    train_model(data_loader, model_save_path, onnx_model_path)
    print("Training complete.")
