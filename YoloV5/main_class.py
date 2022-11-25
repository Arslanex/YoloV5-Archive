import torch
import cv2 as cv
import numpy as np
from time import time

class YoloV5:

    def __init__(self, capture_index, model_name):
        """
        kullanılacak kamera, kullanılacak model  ve modelin çalışacağı cihazın
        atamalarını yapıyoruz

        :param capture_index: kullanılacak kameranın indeksi
        :param model_name: kullanılacak modelin adı
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        """
        Kullanılacak modeli Pytorch üzerinden indiriyor veya kendi modelimizi içe
        aktarıyoruz. Modelinizin içe aktarılabilmesi için bu dosya ile aynı dizinde
        custom.pt isimli bir model olması gerekli

        :return: YoloV5 modeli
        """
        if model_name == "yolov5n":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        elif model_name == "yolov5s":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        elif model_name == "yolov5m":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        elif model_name == "yolov5l":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        elif model_name == "yolov5x":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)

        return model

    def get_video_capture(self):
        """
        Sınıfı oluşturulken belirtilen kamera indeksini kullanarak kameradan görüntü
        alır

        :return: kameradan alınan görüntü
        """
        return cv.VideoCapture(self.capture_index)

    def score_frame(self, frame):
        """
        frame değerini alır ve modele sokrarak çıktılarını alır. Alının çıktıları
        da label ve koordinatlar (cord) olarak ikiye ayırır ve bunları döndürür

        :param frame: kameradan gelen görüntü
        :return: isim etiketleri (label) , bounding box kordinatları (cord)
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        classlarımızı labela dönüştürüyoruz.
        :return: tespit edilen objenin etiketi
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Modelden alınan sonuç değerleri kullanılarak tespit edilen objelerin
        çerceve içine alınması işlemi.

        :param results: modelden dönen sonuçlar
        :param frame: kamerandan gelen görüntü
        :return: tespit edilen cisimlerin çerceveye alındığı görsel
        """

        labels, cord = results
        detection_count = len(labels)   # or len(cord)
        width, height = frame.shape[1], frame.shape[0]

        for i in range(detection_count):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * width), \
                                 int(row[2] * height), int(row[3] * height)

                cv.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                cv.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        return frame

    def __call__(self, *args, **kwargs):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv.resize(frame, (416, 416))

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"her saniye frame yaz : {fps}")

            cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv.imshow('YOLOv5 Detection', frame)

            if cv.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    detector = YoloV5(capture_index=0, model_name='yolov5s')
    detector()