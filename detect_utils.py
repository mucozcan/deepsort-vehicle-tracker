import torchvision.transforms as transforms
import cv2
import numpy as np

"""
SSD example output:
[{'boxes': tensor([[442.5059, 238.3201, 507.2607, 280.1403],
        [171.7978, 255.5709, 227.4966, 295.5227],
        [364.5702, 241.9929, 409.5678, 290.7316],
        ...
        [270.7072, 133.8356, 275.2707, 138.8514]], device='cuda:0',
       grad_fn=<StackBackward>), 
'scores': tensor([0.5262, 0.4450, 0.3461, 0.2393, 0.1884, 0.1600, 0.1473, 0.1453, 0.1361,
        0.1336, 0.1321, 0.1290, 0.1236, 0.1231, 0.1230, 0.1224, 0.1222, 0.1174,
        0.1162, 0.1161, 0.1160, 0.1154, 0.1147, 0.1144, 0.1142, 0.1141, 0.1139,
        ...
        0.0714, 0.0711, 0.0711, 0.0709, 0.0708, 0.0705, 0.0702, 0.0701, 0.0701,
        0.0700, 0.0699], device='cuda:0', grad_fn=<IndexBackward>), 
'labels': tensor([19, 19, 19, 21, 21, 21, 38, 38, 38, 38, 38, 38, 38, 38,  1, 38, 38, 38,
        38, 38, 38, 38, 38, 38, 38, 38, 38,  1, 38, 38, 21, 38, 38, 38,  1, 38,
        38,  1, 38,  1,  1, 38, 38, 38, 38, 38, 38, 38, 38,  1,  1, 38, 38, 19,
        ...
        38, 38, 38, 38, 38, 38, 38, 38,  1, 38, 38, 38,  1,  1,  1, 38, 38, 38,
        19, 38], device='cuda:0')}]
"""

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'cat', 'hair drier', 'toothbrush']


# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# define the torchvision image transforms
transform = transforms.Compose([
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
    
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)


    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.squeeze().tolist()
        box = [xmin, ymin, xmax, ymax]
        # if pred_classes[idx] == 'person':
        dets.append({'coords': box, 'conf': pred_scores[idx], 'label' : pred_classes[idx]})
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
