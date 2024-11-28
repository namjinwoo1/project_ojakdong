cmake_minimum_required(VERSION 3.0.2)
project(project_ojakdong)

## C++ 표준 설정
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## catkin과 OpenCV 찾기
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  robot_state_publisher
  roscpp
  std_msgs
)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4) # OpenCV 4.8 cmake 경로
find_package(OpenCV REQUIRED)
set(DLIB_USE_CUDA OFF)
find_package(dlib REQUIRED PATHS /usr/local/lib/cmake/dlib NO_DEFAULT_PATH)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
set(CMAKE_AUTOMOC ON) # Meta-Object Compiler(MOC)를 자동으로 실행

## Python3 설정
find_package(Python3 REQUIRED COMPONENTS Interpreter)

## catkin 패키지 구성
catkin_package()

## 헤더 및 라이브러리 디렉토리 추가
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

## 실행 파일 생성
add_executable(gui src/gui.cpp src/facetracker.h) 

target_link_libraries(gui
    Qt5::Core
    Qt5::Widgets
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

## userML.py 복사
add_custom_command(TARGET gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/project_ojakdong/src/userML.py ${CMAKE_BINARY_DIR}/userML.py
    COMMENT "Copying userML.py to the build directory"
)

