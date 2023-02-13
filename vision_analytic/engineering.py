from typing import Dict

import cv2
import pandas as pd
import torch
import numpy as np

from vision_analytic.data import CRMProcesor, NotificationManager
from vision_analytic.recognition import FaceRecognition
from vision_analytic.tracking import Tracker
from vision_analytic.utils import (
    xyxy_to_xywh, get_angle, engagement_detect, crop_img, quality_embeddings)
from config.config import (
    DISTANCE_EYES_THRESHOLD, N_EMBEDDINGS, N_EMBEDDINGS_REGISTER, logger
)

class Watchful:
    def __init__(
        self,
        name_vigilant:str,
        recognition: FaceRecognition,
        tracker: Tracker,
        data_manager: CRMProcesor,
        ) -> None:
        
        self.name_vigilant= name_vigilant

        self.recognition = recognition
        self.tracker = tracker
        self.data_manager = data_manager

        self.notification_manager = NotificationManager(data_manager=self.data_manager)
        self.streaming_bbdd = pd.DataFrame(columns=["id_raw", "embedding"])
        self.raw2user_identified = {}
        self.info_user = {}


    def capture(self, source) -> None:

        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det_recognitions = self.recognition.predict(frame, threshold=0.7)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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
                    
                    # update raw2user_identified - add info user
                    for id_raw, result in zip(df_transform["id_raw"], query_resuls):
                        if result["status"]:
                            self.raw2user_identified[id_raw] = result["id_user"]
                            self.info_user[result["id_user"]] = self.data_manager.query_infoclient(
                                result["id_user"]
                                )
                            # generate notifications
                            self.notification_manager.generate_notification(frame, result["id_user"])
                            self.notification_manager.send_sms(result["id_user"])
                            

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
                        objectID = self.info_user[objectID][0]["name"]
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


class CRMRegister(Watchful):
    def __init__(self, name_vigilant: str, recognition: FaceRecognition, data_manager: CRMProcesor) -> None:
        Watchful.__init__(
            self,
            name_vigilant=name_vigilant, 
            recognition=recognition, 
            tracker=None,
            data_manager=data_manager)
        
        self.streaming_bbdd = pd.DataFrame(columns=["id_raw", "embedding", "img_crop"])
        self.state_notification = {}

    def capture(self, source, user_info: Dict) -> None:

        df_embedding_register = None

        cap = cv2.VideoCapture(source)

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        xyxy_crop = self.calculate_crop(W=W, H=H, pxs_x=300, pxs_y=400)

        id_user = user_info["id_user"]
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_img(frame, xyxy_crop)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det_recognitions = self.recognition.predict(img=frame, threshold=0.8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if len(det_recognitions) > 1:
                self.state_notification["more_one_face"] = "more than one face is detected"

            elif len(det_recognitions) == 1:
                face = det_recognitions[0]

                # quality criterial
                quality_aprove = self.quality_criterial(face)
                if quality_aprove:
                    embedding_id = {
                                "id_raw": id_user,
                                "embedding": [face["embedding"]],
                                "img_crop" : [crop_img(frame, face["bbox"])]
                                }
                    self.streaming_bbdd = pd.concat([
                                self.streaming_bbdd,
                                pd.DataFrame.from_dict(embedding_id)
                            ])
                else:
                    self.state_notification["quality_criterial"] = "see the camera or near"

                if len(self.streaming_bbdd) > N_EMBEDDINGS_REGISTER:
                    # quality of embeddings
                    matrix_embedding = np.array(
                        [row for row in self.streaming_bbdd["embedding"].values]
                        )
                    quality_sparce = quality_embeddings(matrix_embedding, threshoold=0.9)

                    if quality_sparce > 0.95:
                        df_embedding_register = self.embedding_transformation(self.streaming_bbdd)

                        self.state_notification["created"] = "Creating register"
                    else:
                        self.state_notification["consistence"] = """
                        not register consistence - restarting
                        """
                        self.streaming_bbdd = pd.DataFrame()

                # draw detection
                start = (int(face["bbox"][0]), int(face["bbox"][1]))
                end = (int(face["bbox"][2]), int(face["bbox"][3]))
                frame = cv2.rectangle(
                    frame,
                    start, 
                    end,
                    color=(255, 0, 0),
                    thickness=2
                    )
            logger.info(self.state_notification)
            self.state_notification = {}
            cv2.imshow("face recognition", frame)

            if df_embedding_register is not None:
                user_info["embedding"] = [df_embedding_register["embedding"].values[0]]
                user_info["meta_data"] = self.streaming_bbdd
                return user_info

            if cv2.waitKey(10) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        

    def calculate_crop(self, W, H, pxs_x: int=300, pxs_y: int=400) -> np.array:

        x_start = int(W / 2) - int(pxs_x/2)
        x_end = int(W / 2) + int(pxs_x/2)
        y_start = int(H / 2) - int(pxs_y/2)
        y_end = int(H / 2) + int(pxs_y/2)

        return np.array([x_start, y_start, x_end, y_end])