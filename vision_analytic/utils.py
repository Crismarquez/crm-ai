from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from config.config import DATA_DIR, TABLES_CONFIG


def to_center_objects(outputs):

    outputs_bbox = outputs[:, :4]
    centroid_x = np.mean([outputs_bbox[:, 0], outputs_bbox[:, 2]], axis=0)
    centroid_y = np.mean([outputs_bbox[:, 1], outputs_bbox[:, 3]], axis=0)
    id_objects = outputs[:, 4]

    center_objects = []

    for id_object, x, y in zip(id_objects, centroid_x, centroid_y):
        center_objects.append((id_object, (int(x), int(y))))

    return center_objects


def xyxy_to_xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def create_bbdd(tables: List = ["user_embeddings", "info_users"]) -> None:

    for table in tables:
        pd.DataFrame(columns=TABLES_CONFIG[table]).to_pickle(DATA_DIR / f"{table}.pkl")


def load_pickle(file_dir: Path):
    with open(file_dir, "rb") as pickle_file:
        content = pickle.load(pickle_file)
    return content

def get_angle(a: np.array, b: np.array, c: np.array) -> float:
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def engagement_detect(left_angle, right_angle, min_angle=80, max_angle=110) -> bool:

    if ((left_angle > min_angle) & (right_angle > min_angle) 
    & (left_angle < max_angle) & (right_angle < max_angle)):
        return True
    else:
        return False


def crop_img(frame, xyxy):
    xyxy = xyxy.astype(int)
    for i, value in enumerate(xyxy):
        if value < 0:
            xyxy[i] = 0

    croped = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    return croped

def quality_embeddings(matrix_embedding: np.ndarray, threshoold:float = 0.9) -> float:
    matrix_cosine = cosine_similarity(
        matrix_embedding,
        matrix_embedding
    )

    sparce_similarity = (matrix_cosine < threshoold).sum()
    total_combination = (
        matrix_cosine.shape[0]*matrix_cosine.shape[1]-len(matrix_cosine.shape)
        )

    quality = 1 - sparce_similarity/total_combination

    return quality