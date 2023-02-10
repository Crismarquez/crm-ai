import typer
import cv2
import torch
import numpy as np

from vision_analytic.recognition import FaceRecognition
from vision_analytic.tracking import Tracker
from vision_analytic.utils import xyxy_to_xywh
from vision_analytic.data import CRMProcesor
from config.config import EMBEDDING_DIMENSION


# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def facerecognition(source=None):
    """
    show face recognition
    """
    model = FaceRecognition()

    if source is None:
        source = 0

    cap = cv2.VideoCapture(source)

    print("bbox for face detected: ")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        prediction, frame = model.predict_draw(frame)

        bbox = [pred["bbox"] for pred in prediction]
        print(bbox, "\n")

        cv2.imshow("face recognition", frame)
        if cv2.waitKey(10) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.command()
def tracking(source=None):

    model = FaceRecognition()
    tracker = Tracker()

    if source is None:
        source = 0

    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        prediction = model.predict(frame)

        if prediction:
            xyxy = []
            confidence = []
            clss = [0] * len(prediction)  # only one class

            for face in prediction:
                xyxy.append(face["bbox"])
                confidence.append(face["det_score"])

            xyxy = torch.tensor(xyxy)
            # get id and centroids dict
            objects = tracker.update(
                xyxy_to_xywh(xyxy), torch.tensor(confidence), torch.tensor(clss), frame
            )

            # draw id
            for objectID, centroid in objects.items():
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


@app.command()
def createregister():

    crm_ddbb = CRMProcesor()

    name = input("Por favor ingrese su nombre: ")
    age = input("Por favor ingrese su edad: ")
    phone = input("Por favor ingrese su numero de teléfono: ")
    id_user = input("Por favor ingrese su numero de identificación: ")
    accept = input("Acepta terminos y condiciones (y/n): ")

    user_info = {
        "name": [name],
        "age": [age],
        "phone": [phone],
        "id_user": [id_user],
        "accept": [accept],
        "embedding": [np.random.rand(EMBEDDING_DIMENSION)],
    }

    result_register = crm_ddbb.create_register(user_info)

    print("*" * 16)
    print("Result of register: ", result_register)

    new_embedding = [np.random.rand(EMBEDDING_DIMENSION)]

    result_query = crm_ddbb.query_embedding(new_embedding, threshold_score=-1.0)

    print("\n", "*" * 16)
    print("Result query embedding (face-match), threshold: -1.0")
    print(result_query)

    print("\n", "*" * 16)
    print("Result query info", "\n")
    for query_user in result_query:
        if query_user["status"]:
            info_query = crm_ddbb.query_infoclient(query_user["id_user"])
            print(info_query)
        else:
            print("not match user")


if __name__ == "__main__":
    app()
