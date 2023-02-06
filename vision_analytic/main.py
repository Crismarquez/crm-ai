import typer
import cv2

from vision_analytic.recognition import FaceRecognition


# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def facerecognition(source=None):
    """
        show face recognition
    """
    model = FaceRecognition()

    if source is None:
        source=0

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
def tracking():
    print("tracking")

if __name__ == "__main__":
    app()