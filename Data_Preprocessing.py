import os
import cv2 as cv

original_dataset_dir = '/home/vincent/data/GliomaImage'

images = cv.imread('/home/vincent/data/GliomaImageProcessing/train/T1CY/T1C.Y.1.png')
print(images.shape)

