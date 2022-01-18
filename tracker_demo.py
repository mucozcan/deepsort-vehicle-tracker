import torch
import torchvision
import numpy as np
import cv2
import sys
from imutils.video import VideoStream
import time

from deepsort import DeepSORT
import detect_utils

def get_gt(dets):

    detections = []
    out_scores = []
    for det in dets:

        coords = det['coords']

        detections.append(coords)
        out_scores.append(det['conf'])

    return detections,out_scores

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

    model.eval().to(device)

    frame_id = 0

    # cap = VideoStream("rtsp:192.168.1.11:8554/test").start()
    cap = VideoStream(0).start()

    sys.path.append("./siamese_network/")
    deepsort = DeepSORT("./siamese_network/ckpts/model640.pt")

    while True:
        start_time = time.time()
        frame = cap.read()

        dets = detect_utils.predict(frame, model, device, 0.6)

        if len(dets) != 0:
            detections,out_scores = get_gt(dets)

            detections = np.array(detections)
            out_scores = np.array(out_scores) 
            tracker,detections_class = deepsort.run(frame,out_scores,detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                #Draw bbox from tracker.
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
                cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                #Draw bbox from detector. Just to compare.
                for det in detections_class:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"FPS: {1/(time.time() - start_time)}")
        frame_id+=1
