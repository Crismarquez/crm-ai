from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Dict
import cv2

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from vision_analytic.main import watchful
from config import config
from config.config import logger

# Define application
app = FastAPI(
    title="crm-vision",
    description="Facial recognition service.",
    version="0.1",
)

@app.get("/video")
async def get_video(camera_source: str):
    
    camera_source = int(camera_source)
    watchful(camera_source)

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start the flask app
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)