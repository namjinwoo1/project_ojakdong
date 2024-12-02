cmake_minimum_required(VERSION 3.0.2)
project(project_ojakdong)

## 컴파일러 설정
set(CMAKE_CXX_STANDARD 14)

## Catkin 패키지 설정
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  roslib
)
find_package(jsoncpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

## Qt 설정
set(CMAKE_AUTOMOC ON) # MOC 처리 자동화
set(CMAKE_AUTORCC ON) # Qt 리소스 자동화 (선택)
set(CMAKE_AUTOUIC ON) # UI 파일 자동화 (선택)

## Catkin 패키지 설정
catkin_package()

## Include 디렉토리 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src # 헤더 파일이 src 폴더에 위치
  /usr/include/jsoncpp
)

## 라이브러리 생성
add_library(${PROJECT_NAME}_lib
    src/checkuser.cpp
    src/labeling.cpp
    src/checkuser.h # src 내 헤더 파일 추가
    src/labeling.h
)

## 실행 파일 생성
add_executable(check_ui
    src/check_ui.cpp
    src/checkuser.cpp
    src/labeling.cpp
)

add_executable(make_config_node src/makeconfig.cpp)
target_link_libraries(make_config_node
  ${catkin_LIBRARIES}
  Qt5::Widgets
  jsoncpp # JSON 라이브러리 연결
)

## 라이브러리 링크
target_link_libraries(check_ui
  ${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Qt5::Widgets
)
