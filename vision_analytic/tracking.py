from pathlib import Path
from typing import Dict

import numpy as np
import torch

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

from config.config import CONFIG_DIR, MODELS_DIR
from vision_analytic.utils import to_center_objects


ALLOWED_TRACKER_WEIGHTS = ["osnet_x0_25_msmt17.pt"]


class Tracker:
    """
    Using bbox, confidence and class this model can predict a unique id object and
    then re-identify the same object and asign the same id in the next frame.

    Methods
    -------
    update (xywh, confidences, clss, frame)
        this method is used to predict the id, return a dictionary whith id and center.
    """

    def __init__(self, tracker_weights="osnet_x0_25_msmt17.pt"):

        self.config_strongsort = CONFIG_DIR / "strong_sort.yaml"
        self.strong_sort_weights = Path(MODELS_DIR, "strong_sort", tracker_weights)

        self.load_tracker()

    def load_tracker(self):
        cfg = get_config()
        cfg.merge_from_file(self.config_strongsort)

        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = "cpu"

        self.strong_sort = StrongSORT(
            self.strong_sort_weights,
            self.device,
            fp16=False,
            max_dist=cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=cfg.STRONGSORT.MAX_AGE,
            n_init=cfg.STRONGSORT.N_INIT,
            nn_budget=cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
        )

    def update(
        self,
        xywh: torch.Tensor,
        confidences: torch.Tensor,
        clss: torch.Tensor,
        frame: np.ndarray,
    ) -> Dict:
        """predict id object and return a dictionary with id and center

        Args:
            xywh (np.ndarray): bbox in detection
            confidences (np.ndarray): confidence of detection
            clss (np.ndarray): class of detection
            frame (np.ndarray): original image

        Returns:
            Dict: {id_pbject, centroid}
        """
        outputs = self.strong_sort.update(
            xywh.cpu(), confidences.cpu(), clss.cpu(), frame
        )

        center_objects = {}
        if len(outputs):
            center_objects = to_center_objects(outputs)

        return center_objects
