cmake_minimum_required(VERSION 3.0.2)
project(project_ojakdong)

## 컴파일러 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

## Catkin 패키지 설정
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  roslib
  sensor_msgs
  cv_bridge
  message_generation
  image_transport
  std_srvs
)

find_package(CUDA REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(OpenCV 4.2 REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

## Darknet 경로 설정
set(DARKNET_PATH ${CMAKE_CURRENT_SOURCE_DIR}/darknet)
include_directories(${DARKNET_PATH}/include)
link_directories(${DARKNET_PATH})

## Qt 설정
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

## 메시지 파일 설정
add_message_files(
  FILES
  DetectionResult.msg
  ClassifiedResult.msg 
)

## 메시지 생성
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

## Catkin 패키지 설정
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs rospy message_runtime cv_bridge std_srvs
)

## Include 디렉토리 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  /usr/include/jsoncpp
  ${CUDA_INCLUDE_DIRS}
)

## 라이브러리 생성
add_library(${PROJECT_NAME}_lib
    src/checkuser.cpp
    src/labeling.cpp
    src/checkuser.h
    src/labeling.h
    src/check_ui.h
)

## 실행 파일 생성
add_executable(check_ui
    src/check_ui.cpp
    src/checkuser.cpp
    src/labeling.cpp
)
target_link_libraries(check_ui
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Qt5::Widgets
)

# add_executable(camera_processor
#     src/camera_processor_node.cpp
# )
# target_link_libraries(camera_processor
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   ${DARKNET_PATH}/libdarknet.so
#   jsoncpp
#   ${CUDA_LIBRARIES}
#   adjust_lib 
# )
add_executable(make_config_node src/makeconfig.cpp)
target_link_libraries(make_config_node
  ${catkin_LIBRARIES}
  Qt5::Widgets
  jsoncpp
)
add_library(adjust_lib
    src/Adjust.cpp
    src/Adjust.hpp
)
target_link_libraries(adjust_lib
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Qt5::Widgets
)
add_executable(filter_node
    src/Adjust.cpp  # Adjust와 관련된 ROS 노드 (추가된 노드)
    src/AdjustNode.cpp
)
target_link_libraries(filter_node
    adjust_lib          # Adjust 라이브러리 링크
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    jsoncpp  
)
# 새로운 check_web_ui 노드 추가
add_executable(check_web_ui
  src/check_web_ui.cpp
  src/checkuser_web.cpp
  src/labeling_web.cpp
  src/DetectionProcessor.cpp
  src/make_dataset.cpp
)
target_link_libraries(check_web_ui
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Python 노드 설치 설정
catkin_install_python(PROGRAMS
    src/finetuning.py
    src/classify.py
    src/camera_feed.py        # 추가된 Python 파일
    src/p_control.py          # 추가된 Python 파일
    src/camera_processor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


