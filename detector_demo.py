import torch
import torchvision
import cv2
import time

import detect_utils

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    #     pretrained=True)

    # model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    # model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        pretrained=True)
    model.eval().to(device)

    frame_id = 0

    cap = cv2.VideoCapture("test.mp4")

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        dets = detect_utils.predict(frame, model, device, 0.7, (360, 640))

        if len(dets) != 0:
            frame = detect_utils.draw_boxes(dets, frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"FPS: {1/(time.time() - start_time)}")
        frame_id += 1
