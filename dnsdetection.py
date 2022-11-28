import ssl
import cv2
import torch

ssl._create_default_https_context = ssl._create_unverified_context

"""
developed by github id : tayyabmd / tayyabmohammedrq@gmail.com
"""

def detecting(frame, model):
    frame = [frame]
    res = model(frame)
    labels, cords = res.xyxyn[0][:, -1], res.xyxyn[0][:, :-1]
    return labels, cords


def color_box(results, frame, classes, acc=0.45):
    labels, cords = results
    n = len(labels)
    x_window, y_window = frame.shape[1], frame.shape[0]

    for i in range(n):
        cords_list = cords[i]
        if cords_list[4] >= acc:
            x1 = int(cords_list[0] * x_window)
            y1 = int(cords_list[1] * y_window)
            x2 = int(cords_list[2] * x_window)
            y2 = int(cords_list[3] * y_window)
            text_d = classes[int(labels[i])]
            if text_d == "DNS":
                color = (255, 0, 0)
            elif text_d == "cavity":
                color = (255, 170, 0)
            elif text_d == "dog":
                color = (0, 170, 127)
            else:
                color = (85, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # object box
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)  # text box
            cv2.putText(frame, text_d + f" {round(float(cords_list[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)  # text adding

    return frame


mydic = {}
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='best.pt file path',
                       force_reload=True)
classes = model.names


def chart_recognisation(ret, frame):
    acc = 0.45
    if ret != 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detecting(frame, model=model)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = color_box(res, frame, classes=classes, acc=acc)
        cv2.imshow('Support and Resistance Detection', frame)
