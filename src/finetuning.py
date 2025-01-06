#!/usr/bin/env python3

import rospy
import rospkg
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import json
from tensorflow.keras.callbacks import Callback
from std_msgs.msg import String

class TrainingProgressCallback(Callback):
    def __init__(self, total_epochs, publisher):
        super(TrainingProgressCallback, self).__init__()
        self.total_epochs = total_epochs
        self.publisher = publisher

    def on_epoch_end(self, epoch, logs=None):
        progress_message = f"Epoch {epoch + 1}/{self.total_epochs} completed."
        rospy.loginfo(progress_message)
        self.publisher.publish(progress_message)


def train_finetuning(train_dir, val_dir, model_save_path, class_indices_path):
    try:
        # TensorFlow 경고 억제 (TensorRT 관련 경고 제거)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # GPU 설정 (GPU 활성화)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # 데이터 전처리 설정
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)

        # 데이터셋 검사
        if len(os.listdir(train_dir)) == 0 or len(os.listdir(val_dir)) == 0:
            rospy.logerr("Dataset directory is empty. Please check your dataset.")
            return

        # 데이터 생성기 설정
        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(
            val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

        # 클래스 일치 여부 확인
        if train_generator.class_indices != val_generator.class_indices:
            rospy.logerr("Train and validation classes do not match.")
            rospy.loginfo(f"Train classes: {train_generator.class_indices}")
            rospy.loginfo(f"Validation classes: {val_generator.class_indices}")
            return

        # 모델 초기화 디버깅
        rospy.loginfo("Initializing MobileNetV2 model...")
        try:
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            rospy.loginfo("MobileNetV2 model initialized successfully.")
        except Exception as e:
            rospy.logerr(f"Error during MobileNetV2 initialization: {e}")
            raise
        
        # 모델 구성
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)  # Dense 레이어 크기 축소
        num_classes = train_generator.num_classes
        rospy.loginfo(f"Number of classes: {num_classes}")
        
        # 최종 출력 레이어
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # 기본 모델 가중치 고정
        for layer in base_model.layers:
            layer.trainable = False

        # 모델 컴파일
        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        rospy.loginfo("Training started...")
        
        # ROS 퍼블리셔 추가
        progress_pub = rospy.Publisher('/finetuning_progress', String, queue_size=10)

        # 콜백 생성
        progress_callback = TrainingProgressCallback(total_epochs=20, publisher=progress_pub)

        # 모델 훈련
        history = model.fit(
            train_generator,
            epochs=20,
            validation_data=val_generator,
            steps_per_epoch=max(train_generator.samples // train_generator.batch_size, 1),
            validation_steps=max(val_generator.samples // val_generator.batch_size, 1),
            callbacks=[progress_callback]  # 콜백 추가
        )

        # 클래스 인덱스 처리
        class_indices = train_generator.class_indices.copy()
        rospy.loginfo(f"Original class indices: {class_indices}")

        # "test" 클래스 제거 후 저장
        if "test" in class_indices:
            del class_indices["test"]

        # 클래스 인덱스 저장
        with open(class_indices_path, 'w') as f:
            json.dump(class_indices, f)
        rospy.loginfo(f"Filtered class indices saved at {class_indices_path}")

        # 모델 저장
        model.save(model_save_path)
        rospy.loginfo(f"Model saved at {model_save_path}")

    except Exception as e:
        rospy.logerr(f"Error during training: {e}")
        raise


def main():
    rospy.init_node('finetuning_node')

    # 패키지 경로 가져오기
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('project_ojakdong')

    # 데이터셋 경로 및 저장 경로 설정
    train_dir = os.path.join(package_path, 'dataset/train')
    val_dir = os.path.join(package_path, 'dataset/val')
    model_save_path = os.path.join(package_path, 'model/mobilenetv2_finetuned.h5')
    class_indices_path = os.path.join(package_path, 'model/class_indices.json')

    # 경로 확인
    if not os.path.exists(train_dir):
        rospy.logerr(f"Training directory not found: {train_dir}")
        return
    if not os.path.exists(val_dir):
        rospy.logerr(f"Validation directory not found: {val_dir}")
        return

    # 학습 함수 호출
    try:
        train_finetuning(train_dir, val_dir, model_save_path, class_indices_path)
    except Exception as e:
        rospy.logerr(f"Error occurred: {e}")


if __name__ == '__main__':
    main()
