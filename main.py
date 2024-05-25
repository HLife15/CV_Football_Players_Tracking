import cv2
import numpy as np

# 파일 경로 설정
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
names_path = 'coco.names'
video_path = 'example.mp4'

# YOLO 모델 로드
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = []
unconnected_layers = net.getUnconnectedOutLayers()
if unconnected_layers.any():
    for i in unconnected_layers.flatten():
        output_layers.append(layer_names[i - 1])

# 클래스 이름 로드
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# tvmonitor 클래스의 인덱스 확인
tvmonitor_index = classes.index('tvmonitor')

# 영상 로드
video = cv2.VideoCapture(video_path)

# 개체 식별을 위한 변수 초기화
object_ids = {}
id_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 영상 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 각각의 검출된 객체에 대해
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id != tvmonitor_index:  # tvmonitor 클래스 제외
                # 객체의 중심좌표 및 박스 크기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 박스의 좌측상단 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 비최대 억제 적용
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 각 개체에 대해 박스 그리기
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # 개체 식별
            if label not in object_ids:
                object_ids[label] = id_count
                id_count += 1
            object_id = object_ids[label]

            # 개체 식별자를 이용하여 색상 결정
            color = (255, 0, 0)  # Blue
            if object_id == 0:
                color = (0, 255, 0)  # Green
            elif object_id == 1:
                color = (0, 0, 255)  # Red

            # 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
