import torch
import cv2 as cv


img = cv.imread("test_image_1.jpeg")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes = model.names
device = 'cuda'

img = cv.resize(img, (640,640))

model.to(device)
results = model(img)

def center_point(x1,y1,x2,y2):
    center_x = int((x1+x2)/2)
    center_y = int((y1+y2)/2)

    return center_x, center_y

def plot_center(center_x, center_y, frame):
    cv.circle(frame, (center_x, center_y), 3, (0,0,255), -1)
    cv.line(frame, (center_x-7, center_y-7), (center_x+7, center_y+7), (0,0,255), 1)
    cv.line(frame, (center_x - 7, center_y + 7), (center_x + 7, center_y - 7), (0, 0, 255), 1)

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
            x,y = center_point(x1,y1,x2,y2)
            label, x0, y0 = screen_position(x,y, labels[i])
            plot_center(x,y, frame)

            print("\nLabel ::",label)
            print("Center X ::",x, "-", x0, "\nCenter Y ::",y, "-", y0)

def screen_position(center_x, center_y, label_index):
    global classes
    label = classes[int(label_index)]

    x_position = None
    y_position = None

    if 0<=center_x<220:
        x_position="L"
        if 0 <= center_y < 220:
            y_position="T"
        elif 220 <= center_y < 440:
            y_position="C"
        elif 440 <= center_y <= 660:
            y_position="B"

    elif 220 <= center_x < 440:
        x_position="C"
        if 0 <= center_y < 220:
            y_position = "T"
        elif 220 <= center_y < 440:
            y_position = "C"
        elif 440 <= center_y <= 660:
            y_position = "B"

    elif 440<= center_x <= 660:
        x_position="R"
        if 0 <= center_y < 220:
            y_position = "T"
        elif 220 <= center_y < 440:
            y_position = "C"
        elif 440 <= center_y <= 660:
            y_position = "B"

    return label, x_position, y_position

def screen_lines(frame):
    x0, y0 = 0, 0
    x1, y1 = 220, 220
    x2, y2 = 440, 440
    x3, y3 = 660, 660

    h = 20

    cv.line(frame, (x1,y0), (x1,y3), (255,255,255), 2)
    cv.line(frame, (x2, y0), (x2, y3), (255, 255, 255), 2)
    cv.line(frame, (x0, y1), (x3, y1), (255, 255, 255), 2)
    cv.line(frame, (x0, y2), (x3, y2), (255, 255, 255), 2)

    cv.putText(frame, ("T / L"), (int((x0+x1)/2) -25 ,h), cv.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
    cv.putText(frame, ("T / C"), (int((x1+x2)/2) -25 ,h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv.putText(frame, ("T / R"), (int((x2+x3)/2) -25 ,h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

    cv.putText(frame, ("C / L "), (int((x0+x1)/2)-25, y1+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv.putText(frame, ("C / C "), (int((x1+x2)/2)-25, y1+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv.putText(frame, ("C / R "), (int((x2+x3)/2)-25, y1+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

    cv.putText(frame, ("B / L "), (int((x0+x1)/2)-25, y2+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv.putText(frame, ("B / C "), (int((x1+x2)/2)-25, y2+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    cv.putText(frame, ("B / R "), (int((x2+x3)/2)-25, y2+h), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    return frame

plot_rectangles(img, results)
img = screen_lines(img)

cv.imshow('YOLOv5 Detection', img)
cv.waitKey(0)
cv.destroyAllWindows()