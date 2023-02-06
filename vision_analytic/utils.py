import numpy as np
import torch

def to_center_objects(outputs):

    outputs_bbox = outputs[:, :4]
    centroid_x = np.mean([outputs_bbox[:, 0], outputs_bbox[:, 2]], axis=0)
    centroid_y = np.mean([outputs_bbox[:, 1], outputs_bbox[:, 3]], axis=0)
    id_objects = outputs[:, 4]

    center_objects = {}

    for id_object, x, y in zip(id_objects, centroid_x, centroid_y):
        center_objects[id_object] = [int(x), int(y)]
    
    return center_objects

def xyxy_to_xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y