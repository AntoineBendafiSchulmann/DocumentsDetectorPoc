import os
import torch
from yolov5 import train

if __name__ == '__main__':
    dataset_yaml = '../dataset/dataset.yaml'
    pretrained_weights = 'yolov5s.pt'

    train.run(
    data=dataset_yaml, 
    weights='yolov5s.pt',
    epochs=50,   
    batch_size=16,  
    img_size=640, 
    name='document_detector_yolov5'
)
