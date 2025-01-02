std_msgs/Header header
sensor_msgs/Image image          # 검출된 이미지
int32[] class_ids                # 검출된 객체의 클래스 ID 배열
float32[] confidences            # 검출된 객체의 신뢰도 배열
geometry_msgs/Polygon[] boxes    # 검출된 객체의 바운딩 박스 배열 (좌표)
geometry_msgs/Point32[] centers  # 검출된 객체의 중심 좌표 배열 (추가된 부분)
