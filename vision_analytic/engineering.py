from typing import Dict

import cv2
import pandas as pd
import torch

from vision_analytic.data import CRMProcesor
from vision_analytic.recognition import FaceRecognition
from vision_analytic.tracking import Tracker
from vision_analytic.utils import xyxy_to_xywh, get_angle, engagement_detect
from config.config import DISTANCE_EYES_THRESHOLD, N_EMBEDDINGS

class Watchful:
    def __init__(
        self,
        name_vigilant:str,
        recognition: FaceRecognition,
        tracker: Tracker,
        data_manager: CRMProcesor
        ) -> None:
        
        self.name_vigilant= name_vigilant

        self.recognition = recognition
        self.tracker = tracker
        self.data_manager = data_manager

        self.streaming_bbdd = pd.DataFrame(columns=["id_raw", "embedding"])
        self.raw2user_identified = {}


    def capture(self, source) -> None:

        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            det_recognitions = self.recognition.predict(frame, threshold=0.7)
            if det_recognitions:
                xyxy = []
                confidence = []
                clss = [0] * len(det_recognitions)  # only one class

                for face in det_recognitions:
                    xyxy.append(face["bbox"])
                    confidence.append(face["det_score"])

                xyxy = torch.tensor(xyxy)
                # get id and centroids dict
                objects = self.tracker.update(
                    xyxy_to_xywh(xyxy), torch.tensor(confidence), torch.tensor(clss), frame
                )

                # det_recognitions + tracking
                faces_metadata = []
                for face, info_tracking in zip(det_recognitions, objects):
                    face["id_raw_info"] = {
                        "id_raw": info_tracking[0],
                        "center": info_tracking[1]
                        }
                    faces_metadata.append(face)

                # det_recognitions + tracking + user_id
                for face in faces_metadata:
                    raw2user_info = self.raw2user_id(face["id_raw_info"]["id_raw"])
                    face["raw2user_info"] = raw2user_info

                # add streaming  if quality criterial is ok
                for face in faces_metadata:
                    if not face["raw2user_info"]["raw2user_status"]:
                        quality_aprove = self.quality_criterial(face)
                        if quality_aprove:
                            embedding_id = {
                                "id_raw": face["id_raw_info"]["id_raw"],
                                "embedding": [face["embedding"]]
                                }
                            self.streaming_bbdd = pd.concat([
                                self.streaming_bbdd,
                                pd.DataFrame.from_dict(embedding_id)
                            ])

                # embedding transformation
                # select N=5 embedding to generate prediction
                count_values = self.streaming_bbdd["id_raw"].value_counts()
                to_filter_id = list(count_values[count_values>N_EMBEDDINGS].index)
                df_to_predict = self.streaming_bbdd[
                    self.streaming_bbdd["id_raw"].isin(to_filter_id)
                    ]

                if len(df_to_predict)>0:
                    df_transform = self.embedding_transformation(df_to_predict)

                    query_resuls = self.data_manager.query_embedding(
                        [embedding for embedding in df_transform["embedding"].values], 
                        threshold_score=0.7
                        )
                    
                    # update raw2user_identified
                    for id_raw, result in zip(df_transform["id_raw"], query_resuls):
                        if result["status"]:
                            self.raw2user_identified[id_raw] = result["id_user"]

                    # clean stream
                    self.streaming_bbdd = self.streaming_bbdd.drop(
                                    self.streaming_bbdd[self.streaming_bbdd["id_raw"].isin(
                                        list(df_transform["id_raw"]))].index
                                    )


                # draw id
                for face in faces_metadata:
                    object_tracked = face["id_raw_info"]
                    centroid = object_tracked["center"]
                    #query info
                    if face["raw2user_info"]["raw2user_status"]:
                        objectID = face["raw2user_info"]["user_id"]
                    else:
                        objectID = object_tracked["id_raw"]
                    cv2.putText(
                        frame,
                        "ID {}".format(objectID),
                        (centroid[0] - 5, centroid[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

            cv2.imshow("face recognition", frame)
            if cv2.waitKey(10) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


    def quality_criterial(self, face_metadata: Dict) -> bool:

        left_angle = get_angle(
            face_metadata["kps"][0],
            face_metadata["kps"][2],
            face_metadata["kps"][3],
        )
        right_angle = get_angle(
            face_metadata["kps"][1],
            face_metadata["kps"][2],
            face_metadata["kps"][4]
        )

        is_engagement = engagement_detect(
            left_angle=left_angle, 
            right_angle=right_angle,
            min_angle=80, 
            max_angle=110
            )

        distance_eyes = face_metadata["kps"][1][0] - face_metadata["kps"][0][0]

        if is_engagement and distance_eyes > DISTANCE_EYES_THRESHOLD:
            return True
        return False

    def raw2user_id(self, raw_id) -> Dict:
        if raw_id in self.raw2user_identified:
            raw2user_status = True
            user_id = self.raw2user_identified[raw_id]
        else:
            raw2user_status = False
            user_id = 0

        return {"raw2user_status": raw2user_status, "user_id": user_id}


    def embedding_transformation(self, raw_embeddings: pd.DataFrame) -> pd.DataFrame:
        # mean model
        df_predict = raw_embeddings.groupby("id_raw")["embedding"].mean().reset_index()
        return df_predict
