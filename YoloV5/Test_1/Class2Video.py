import torch
import cv2 as cv

cap = cv.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

while cap.isOpened():
    _, main_frame = cap.read()

    model.to(device)
    frame = [main_frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    x_shape, y_shape = main_frame.shape[1], main_frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv.rectangle(main_frame, (x1, y1), (x2, y2), bgr, 2)
            label = classes[int(labels[i])]
            cv.putText(main_frame, label, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

    cv.imshow('YOLOv5 Detection', main_frame)

    if cv.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()