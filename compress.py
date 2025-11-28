import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/home/robot/ultralytics-yolo11-main/runs/train2/RASL-YOLO/weights/best.pt',
        'data':'/home/robot/ultralytics-yolo11-main/data.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 4,
        'workers': 4,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'project':'runs/prune',
        'name':'YOLO11-RASL-lamp-exp1',
        'amp':False,
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.5,
        'reg': 0.01,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
        'iterative_steps': 200,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)