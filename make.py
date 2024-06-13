import os
import sys
from ultralytics import YOLO

# Thank god this works:
# For AMD GPUs:
# https://github.com/ROCm/ROCm/issues/2536
# Comment from "supersonictw"
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

currentDir = os.getcwd()
modelConfig = sys.argv[1]
dataYamlLocation = sys.argv[2]
modelName = sys.argv[3]

model = YOLO(f'{currentDir}/models/configurations/{modelConfig}.yaml')
# model = YOLO('yolov8m.pt')
# Train the model
results = model.train(
    data=currentDir + f'/escooters/dataBank/IMAGES/{dataYamlLocation}/data.yaml',     # path to the data config file
    epochs=20,                                              # number of training epochs
    imgsz=900,                                              # image size
    batch=16,                                               # batch size
    device='0',                                             # specify GPU id (use 'cpu' for CPU training)
    name=f'{modelName}'                                     # name of the run
)
# Evaluate the model
results = model.val()

exportPath = model.export(format='onnx')
with open('exportPath.txt', 'w') as f:
    f.write(exportPath)