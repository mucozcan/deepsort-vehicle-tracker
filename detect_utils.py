import torchvision.transforms as transforms
import cv2
import numpy as np

import config as cfg


# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(cfg.coco_names), 3))

# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])


def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and 
    class labels. 
    """
    dets = []
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)

    pred_classes = [cfg.coco_names[i]
                    for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.squeeze().tolist()
        box = [xmin, ymin, xmax, ymax]
        print(pred_classes[idx])
        if pred_classes[idx] in cfg.classes_to_track:
            dets.append(
                {'coords': box, 'conf': pred_scores[idx], 'label': pred_classes[idx]})
    return dets


def draw_boxes(detections, image):
    """
    Draws the bounding box around a detected object.
    """
    color = (0, 0, 255)
    image = np.asarray(image)
    for i, det in enumerate(detections):
        box = det["coords"]

        cv2.rectangle(
            image,
            (int(box[0]), int(box[1] - 35)),
            (int(box[2]), int(box[3])),
            color, 3
        )
        cv2.putText(image, str(det["label"]), (int(box[0] + 10), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3,
                    lineType=cv2.LINE_AA)
    return image


def get_gt(dets):

    detections = []
    out_scores = []
    for det in dets:

        coords = det['coords']

        detections.append(coords)
        out_scores.append(det['conf'])

    return detections, out_scores
