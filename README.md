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
첨부한 example.mp4는 축구 경기의 일부 장면이 담긴 동영상이고 이 동영상에 OpenCV와 YOLO를 이용해 Tracking을 진행해보겠다.
