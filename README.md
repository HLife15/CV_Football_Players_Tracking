이번 텀프로젝트에서 나는 축구 경기 동영상을 불러와 움직이는 선수들과 심판들을 추적하는 코드를 작성해보았다. 
수업 시간에 배운 내용 중 Object Detection에 가장 흥미를 느끼기도 했고 그렇게 수업 시간에 배운 기술들을 내가 좋아하는 축구에 대입해보고 싶었다. 
Object Detection에는 R-CNN, YOLO, SSD, Retina Net 등 다양한 기술들을 활용할 수 있는데 그 중에 나는 YOLO를 사용해서 구현해보았다. 
<br><br><br>
![Object_detection_illustrated_from_image_recognition_and_localization_704ca34bd8](https://github.com/HLife15/CV_Football_Players_Tracking/assets/162321808/219d34c6-b933-4d12-b7a9-61b82f6e10b8)
<br><br><br>
YOLO(You Only Look Once)는 실시간 객체 감지 알고리즘으로, 이미지나 동영상에서 객체를 식별하고 식별한 객체 주위에 경계 상자를 그리는 기술이다. 
실시간으로 객체를 감지하고, 한번의 예측으로 여러 객체를 동시에 감지할 수 있어 처리 속도가 빠르고 정확도가 높아 다양한 분야에서 폭 넓게 사용되고 있다. 
YOLO를 이용하기 위해서는 YOLO모델의 가중치(yolov3.weights), 구성(yolov3.cfg), 클래스 이름(coco.names) 등을 불러와야 한다. 
yolov3.cfg와 coco.names는 해당 저장소에 업로드했고 yolov3.weights의 경우 용량이 커 업로드할 수 없는 관계로 다운로드 링크를 남긴다. 
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights

<br><br><br>
![example - frame at 0m2s](https://github.com/HLife15/CV_Football_Players_Tracking/assets/162321808/c7e7ea27-6416-4045-a709-978a62491140)
<br><br><br>
첨부한 example.mp4는 축구 경기의 일부 장면이 담긴 동영상이고 이 동영상에 OpenCV와 YOLO를 이용해 Tracking을 진행해보았다. 
(대부분의 코드들은 ChatGPT를 포함한 여러 자료들을 참고하였고 참고자료들은 글 마무리 부분에 적어놓았다.)
<br><br>

```
import cv2
import numpy as np
```

<br>
OpenCV와 수치 연산에 필요한 라이브러리들을 가져온다.
<br><br>

```
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
names_path = 'coco.names'
video_path = 'example.mp4'
```

<br>
YOLO에 필요한 파일들과 동영상 파일을 불러온다.
<br><br>

```
net = cv2.dnn.readNet(weights_path, config_path)
```

<br>
YOLO 모델을 불러온다.
<br><br>

```
layer_names = net.getLayerNames()
output_layers = []
unconnected_layers = net.getUnconnectedOutLayers()
if unconnected_layers.any():
    for i in unconnected_layers.flatten():
        output_layers.append(layer_names[i - 1])
```

<br>
YOLO 모델이 출력될 레이어를 설정한다.
<br><br>

```
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")
```

<br>
coco.names에 적혀있는 클래스 이름들을 리스트로 불러온다. (person, sports ball 등 여기에 적혀있는 클래스 이름들 역시 화면에 출력된다)
<br><br>

```
tvmonitor_index = classes.index('tvmonitor')
```

<br>
YOLO 모델 출력 중에 원치 않는 tvmonitor 클래스가 출력이 되어 그걸 방지하기 위해 tvmonitor 클래스의 인덱스를 받아온다. 삭제 작업은 이후에 진행한다.
<br><br>

```
video = cv2.VideoCapture(video_path)
```

<br>
동영상 파일을 불러온다.
<br><br>

```
object_ids = {}
id_count = 0
```

<br>
각 객체에 할당된 식별자를 저장하는 배열과 식별자를 생성하기 위한 카운터를 초기화한다.
<br><br>

```
while True:
    ret, frame = video.read()
    if not ret:
        break
```

<br>
동영상을 한 프레임씩 처리한다.
<br><br>

```
height, width = frame.shape[:2]
```

<br>
현재 동영상(프레임)의 높이와 너비를 저장한다.
<br><br>

```
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
```

<br>
위에서 저장한 프레임의 크기에 맞게 YOLO 모델을 전처리하고, 모델에 입력하여 객체를 검출한다.
<br><br>

```
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and class_id != tvmonitor_index:  
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
```

<br>
YOLO 모델의 출력에 관한 부분이다. 경계 상자의 좌표와 신뢰도, 클래스 ID 등을 포함한다.
<br><br>

```
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

<br>
중복으로 검출된 객체를 제거한다. 이 코드를 적기 전에는 객체 하나에 두 세개의 경계 상자가 그려지면서 화면이 난잡해지는 경우가 있었다.
<br><br>

```
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        if label not in object_ids:
            object_ids[label] = id_count
            id_count += 1
        object_id = object_ids[label]

        color = (255, 0, 0)  # Blue
        if object_id == 0:
            color = (0, 255, 0)  # Green
        elif object_id == 1:
            color = (0, 0, 255)  # Red

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{label} {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```

<br>
객체에 대해 경계 상자를 그리고 클래스 이름(person 등등)과 객체 식별자(0, 1, 2, ...)를 함께 표시한다.
<br><br>

```
cv2.imshow("Frame", frame)
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
    break
```

<br>
처리된 영상을 화면에 표시하고 'q'키를 누르면 종료한다.
<br><br>

```
video.release()
cv2.destroyAllWindows()
```

<br>
동영상 파일을 닫고 코드 실행을 종료한다.
<br><br><br>
다음과 같은 과정으로 구현한 완성본은 아래와 같다. <br>

![111](https://github.com/HLife15/CV_Football_Players_Tracking/assets/162321808/e3efb2f8-f6ea-478e-b660-3c9498c32467) <br>

<br><br>
원본 동영상에 Tracking이 적용된 화면 <br>



<br><br>
동영상에도 나와 있듯 대부분의 상황에서 Tracking이 잘 진행되었다. 다만 몇몇 장면에선 person, sports ball 이외의 다른 클래스로 판별되기도 했고 
경기장 밖의 안전요원이나 관중들이 Tracking 되는 등 아쉬운 부분들이 있었다. 
그래도 완벽하진 않아도 수업 시간에 배웠던 것을 바탕으로 무언가를 만들어보니 뿌듯했다. 
앞으로도 공부해서 보다 더 정확하고 다양한 기능을 가진 것을 만들어보는 프로젝트를 진행해봐야겠다. <br><br>

[참고 자료]
<br>
https://aggiesportsanalytics.com/projects/soccer-offside-tracker
<br>
https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb
<br>
https://www.reddit.com/r/dataisbeautiful/comments/119szzy/oc_football_players_tracking_with_yolov5/
<br>
https://chatgpt.com/





