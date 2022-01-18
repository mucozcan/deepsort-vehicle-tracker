import torch
import torchvision
import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.deep_sort.detection import Detection
from siamese_network.utils import get_gaussian_mask

class DeepSORT():
    def __init__(self, feature_ext_path):
        
        self.feature_extractor = torch.load(feature_ext_path)
        self.feature_extractor = self.feature_extractor.cuda().eval()

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.5, 100)

        self.tracker = Tracker(self.metric)
        self.gaussian_mask = get_gaussian_mask(image_size=128, aspect_ratio=1).cuda()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
            ])


    def preproc(self, frame, detections):

        objects = []

        for det in detections:

            for i in range(len(det)):
                if det[i] < 0:
                    det[i] = 0

            img_h, img_w, img_ch = frame.shape

            x_min, y_min, w, h = det

            if x_min > img_w:
                x_min = img_w

            if y_min > img_h:
                y_min = img_h

            x_max = x_min + w
            y_max = y_min + h

            x_min = int(x_min)
            y_min = int(y_min)

            x_max = int(x_max)
            y_max  = int(y_max)

           
            try:
                obj = frame[y_min:y_max, x_min:x_max, :]
                obj = self.transforms(obj)
                objects.append(obj)
            except ValueError:
                continue

        objects = torch.stack(objects)

        return objects

    def run(self, frame, out_scores, out_boxes):


        if out_boxes==[]:
            self.tracker.predict()
            print("No detections.")

            trackers = self.tracker.tracks

            return trackers

        detections = np.array(out_boxes)

        processed_objs = self.preproc(frame, detections).cuda()
        processed_objs = processed_objs * self.gaussian_mask

        features = self.feature_extractor.forward_once(processed_objs)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)


        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(detections, out_scores, features)]

        out_boxes = np.array([d.tlwh for d in dets])
    
        out_scores = np.array([d.confidence for d in dets])
        indices = prep.non_max_suppression(out_boxes, 0.8, out_scores)

        dets = [dets[i] for i in indices]
        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets




