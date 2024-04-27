import os
import time
from ultralytics import YOLO
import cv2
import numpy as np

images_dir = os.path.join('.', 'Images')
model_path = os.path.join('.',  'runs', 'detect', 'train4', 'weights', 'last.pt')
# Load a model
model = YOLO(model_path)  # load a custom model

# Load image
new_image_path = os.path.join(images_dir, '5_5.png')
img = cv2.imread(new_image_path)

# Perform detection
result = model(img)

# Save the image with boxes
image_path_out = '{}.out.jpg'.format(new_image_path)
cv2.imwrite(image_path_out, img)

# Display the image with boxes
cv2.imshow('Image with boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
