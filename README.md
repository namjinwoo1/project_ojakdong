# 팀 프로젝트 오작동

- 프로젝트 소개

사용자를 추적하는 촬영 로봇 개발

    

- 개발 기간

2개월


- 개발 기능

Yolov8n을 사용해서 사람을 검출하고 검출한 사람을 데이터셋으로 만드는 기능

만든 데이터셋을 MobileNetV2를 사용해서 사용자와 비사용자를 구별하는 파인튜닝 모델 학습 기능

구별된 정보를 통해 필터링처리하고 처리된 화면을 녹화하면서 GUI에 표현하는 기능

로봇이 사용자를 추적하고 이동하는 기능


# 테스트 사양

Ubuntu 20.04

ROS-Noetic

CUDA 11.8

Tensorflow 2.12

TensorRT 8.4.3

# 테스트 영상
사람 탐지, 사용자 등록 및 모델생성

https://youtube.com/shorts/cA0NUR375go

화면 출력 및 사용자 추적

https://youtube.com/shorts/MEqfHe9M2nk


