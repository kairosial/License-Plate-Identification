import os
from roboflow import Roboflow

rf = Roboflow(api_key="api_key")
project = rf.workspace("yolov8-7lsof").project("licenseplatedetection-bcet9")
version = project.version(1)
dataset = version.download("yolov8")