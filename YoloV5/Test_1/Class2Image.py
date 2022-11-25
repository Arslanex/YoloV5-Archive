import torch
import cv2 as cv

pic = cv.imread("test_image_1.jpeg")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pic = cv.resize(pic, (500,500))

model.to(device)
img = [pic]
results = model(img)


labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

n = len(labels)

x_shape, y_shape = pic.shape[1], pic.shape[0]

for i in range(n):
    row = cord[i]
    print(row)
    if row[4] >= 0.5:
        x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), \
                         int(row[2] * x_shape), int(row[3] * y_shape)

        bgr = (0, 255, 0)
        cv.rectangle(pic, (x1, y1), (x2, y2), bgr, 2)

cv.imshow('YOLOv5 Detection', pic)
cv.waitKey(0)
cv.destroyAllWindows()