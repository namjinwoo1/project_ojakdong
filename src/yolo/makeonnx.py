from ultralytics import YOLO

# YOLOv8n 모델 불러오기
model = YOLO('./yolov8n.pt')

# ONNX 파일로 변환
model.export(format='onnx', dynamic=False, imgsz=(416, 416), opset=11)

# 변환된 ONNX 파일을 원하는 위치로 이동
import shutil
shutil.move('yolov8n.onnx', '../../model/yolov8n.onnx')
