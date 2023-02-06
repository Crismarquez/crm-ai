from typing import List, Tuple

import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceRecognition:
    """
        Contain models for face-detec, face-embbeding, age, gender, pose.
        args:
             providers: list using GPU or CPU
             allowed_modules: list allow models for detec, ambedding, gender, etc.
             det_size: tuple set the size for input image

        Methods
        -------
        predict (img: np.ndarray, threshold: float = 0.7)
            take the image and produce list of dictionaries with feature predicts
            according to allowed_modules.
            the 'threshold' parameter is used for face detection score

    """
    def __init__(
        self,
        providers: List = ['CPUExecutionProvider'],
        allowed_modules: List = ["detection", "recognition"],
        det_size: Tuple = (320, 320)
        ) -> None:
        
        self.providers = providers
        self.allowed_modules = allowed_modules
        self.det_size = det_size

        self.load_model()


    def load_model(self):
        self.model = FaceAnalysis(
            providers=self.providers,
            allowed_modules = self.allowed_modules
            )
        self.model.prepare(ctx_id=0, det_size=self.det_size)


    def _process(self, img: np.ndarray):
        """raw function to use insightface prediction on a image

        Args:
            img (np.ndarray): frame to detect faces

        Returns:
            List[Dict]: List with each face detected, and the dict contains, (depend of modules): 
                'bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose',
                'landmark_2d_106', 'gender', 'age', 'embedding'
                """

        faces = self.model.get(img)
        return faces

    def _clean_output(self, faces: List):

        clean_faces = []
        for face in faces:
            clean_faces.append({key: value for key, value in face.items()})

        return clean_faces


    def predict(self, img: np.ndarray, threshold: float = 0.7):
        faces = self._process(img)

        faces_filtered = [face for face in faces if face["det_score"] > threshold]
        faces_filtered = self._clean_output(faces_filtered)

        return faces_filtered

    
    def predict_draw(self, img: np.ndarray, threshold: float = 0.7):
        faces = self._process(img)

        img_draw = self.model.draw_on(img, faces)

        faces_filtered = [face for face in faces if face["det_score"] > threshold]
        faces_filtered = self._clean_output(faces_filtered)


        return faces_filtered, img_draw


