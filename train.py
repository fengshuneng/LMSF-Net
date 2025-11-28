import warnings, os
import glob
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0"     

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('./RASF-YOLO.yaml')
    model.train(data='./data.yaml',
                cache=False,
                imgsz=640,  
                epochs=200,  
                batch=4,  
                close_mosaic=10, 
                workers=4, 
                device='0', 
                optimizer='SGD', 
                lr0=0.01,  
                weight_decay=0.0005,  
                momentum=0.937,  
                # patience=0, 
                # resume=True, 
                amp=False,  
                # fraction=0.2,
                project='runs/train',
                name='exp', 
                )
