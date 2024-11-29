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

find_package(OpenCV REQUIRED)

set(CMAKE_AUTOMOC ON) # MOC 처리 자동화
set(CMAKE_AUTORCC ON) # Qt 리소스 자동화 (선택)
set(CMAKE_AUTOUIC ON) # UI 파일 자동화 (선택)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

## Catkin 패키지 설정
catkin_package()

## Include 디렉토리 설정
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Qt5Widgets_INCLUDE_DIRS}
)

# add_executable(gui_node src/gui.cpp src/main.cpp)
# target_link_libraries(gui_node
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   Qt5::Widgets
# )

## 실행 파일 생성
add_executable(check_ui src/check_ui.cpp src/checkuser.cpp)

## 라이브러리 링크
target_link_libraries(check_ui
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Qt5::Widgets
)

