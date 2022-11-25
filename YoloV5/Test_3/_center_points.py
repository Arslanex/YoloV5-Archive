import torch
import cv2 as cv

def center_point(x1,y1,x2,y2, frame):
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)

    cv.circle(frame, (center_x, center_y), 3, (0,0,255), -1)
    cv.line(frame, (center_x-7, center_y-7), (center_x+7, center_y+7), (0,0,255), 1)
    cv.line(frame, (center_x - 7, center_y + 7), (center_x + 7, center_y - 7), (0, 0, 255), 1)

    return center_x, center_y

def target_rectangle(x1,y1,x2,y2, frame):
    cv.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 1)
    cv.line(frame, (int((x1+x2)/2), y1), (int((x1+x2)/2), y2), (255,0,255), 1)
    cv.line(frame, (x1,int((y1 + y2) / 2)), (x2,int((y1 + y2) / 2)), (255,0,255), 1)

def plot_rectangles(frame, results):
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.1:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), \
                             int(row[2] * x_shape), int(row[3] * y_shape)

        target_rectangle(x1,y1,x2,y2,frame)
        x,y = center_point(x1,y1,x2,y2,frame)
        print("\nCenter X ::",x, "\nCenter Y ::",y)


pic = cv.imread("test_image_1.jpeg")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
classes = model.names
device = 'cuda'

pic = cv.resize(pic, (660,660))

model.to(device)
img = [pic]
results = model(img)

plot_rectangles(pic, results)

cv.imshow('YOLOv5 Detection', pic)
cv.imwrite("RESULT_1.png", pic)

cv.waitKey(0)
cv.destroyAllWindows()