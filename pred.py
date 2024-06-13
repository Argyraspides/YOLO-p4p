import os

from ultralytics import YOLO
# Thank god this works:
# For AMD GPUs:
# https://github.com/ROCm/ROCm/issues/2536
# Comment from "supersonictw"
from os import putenv

putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

with open('exportPath.txt', 'r') as f:
    bestModelPath = f.read().strip()

modelPath = os.path.join(os.getcwd(), bestModelPath.replace(".onnx", ".pt"))
model = YOLO(modelPath)

testFolderPath = os.path.join(os.getcwd(), 'escooters', 'test', 'images')
outputFolderPath = os.path.join(os.getcwd(), 'escooters', 'predictions')

os.makedirs(outputFolderPath, exist_ok=True)  # Create the output folder if it doesn't exist

imagePaths = [os.path.join(testFolderPath, f) for f in os.listdir(testFolderPath)
              if f.endswith(('.jpg', '.png', '.jpeg'))]

# Run inference and save results with labels
results = model(imagePaths)

i = 0
for result in results:
    # result.show()  # display to screen
    result.save(filename=f"{os.getcwd()}/escooters/predictions/result{i}.jpg")  # save to disk
    i += 1