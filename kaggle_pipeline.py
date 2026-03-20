"""
Leonardo - Airborne Object Recognition Challenge
1st Place Efficiency Strategy Pipeline for Kaggle

Optimized for:
- 41% Tiny Objects (imgsz=1280)
- Max 359 Objects per frame (max_det=500)
- 30% Infrared cameras (gray=0.3, hsv_s=0.5)
- 40% Motion blur (mosaic, mixup, blur augmentations)
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from glob import glob

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
# Disable wandb login prompt for offline Kaggle capacity
os.environ["WANDB_DISABLED"] = "true"

TRAIN_IMG_DIR = '/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge/train'
TRAIN_CSV = '/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge/train.csv'
TEST_IMG_DIR = '/kaggle/input/competitions/leonardo-airborne-object-recognition-challenge/test'
YOLO_ROOT = '/kaggle/working/yolo_data'

CLASSES = sorted(['Aircraft', 'Drone', 'GroundVehicle', 'Helicopter', 'Human', 'Obstacle', 'Ship'])
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def convert_dataset():
    """
    Converts Kaggle format (x_min, y_min, x_max, y_max) to YOLO format 
    (x_center, y_center, width, height) and uses symlinks to save disk space.
    """
    if not os.path.exists(TRAIN_CSV):
        print(f"Dataset not found at {TRAIN_CSV}. Skipping conversion.")
        return

    print("Converting dataset to YOLO format...")
    df = pd.read_csv(TRAIN_CSV)
    df[['x_min', 'y_min', 'x_max', 'y_max']] = df['bbox'].str.split(' ', expand=True).astype(float)

    df['w'] = df['x_max'] - df['x_min']
    df['h'] = df['y_max'] - df['y_min']
    df['x_center'] = df['x_min'] + (df['w'] / 2)
    df['y_center'] = df['y_min'] + (df['h'] / 2)
    df['class_id'] = df['class'].map(CLASS_TO_ID)

    # 80-20 train/val split safely based on Unique Image IDs to avoid data leakage
    image_ids = df['ImageId'].unique()
    np.random.seed(42)
    np.random.shuffle(image_ids)
    split_idx = int(len(image_ids) * 0.8)
    train_ids, val_ids = set(image_ids[:split_idx]), set(image_ids[split_idx:])

    for split in ['train', 'val']:
        os.makedirs(f"{YOLO_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_ROOT}/labels/{split}", exist_ok=True)

    def process_split(split_name, ids_set):
        split_df = df[df['ImageId'].isin(ids_set)]
        grouped = split_df.groupby('ImageId')
        
        for img_id, group in tqdm(grouped, desc=f"Linking {split_name} split"):
            src_img = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
            dst_img = os.path.join(YOLO_ROOT, 'images', split_name, f"{img_id}.png")
            
            # Symlinks bypass the 20GB disk limit
            if not os.path.exists(dst_img) and os.path.exists(src_img):
                os.symlink(src_img, dst_img)
            
            label_content = []
            for _, row in group.iterrows():
                if pd.notna(row['class_id']):
                    line = f"{int(row['class_id'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['w']:.6f} {row['h']:.6f}"
                    label_content.append(line)
            
            label_path = os.path.join(YOLO_ROOT, 'labels', split_name, f"{img_id}.txt")
            with open(label_path, 'w') as f:
                f.write("\n".join(label_content))

    process_split('train', train_ids)
    process_split('val', val_ids)
    print("Data conversion complete!")

def create_yaml():
    yaml_content = f"""
path: {YOLO_ROOT}
train: images/train
val: images/val

names:
  0: Aircraft
  1: Drone
  2: GroundVehicle
  3: Helicopter
  4: Human
  5: Obstacle
  6: Ship
"""
    with open('/kaggle/working/data.yaml', 'w') as f:
        f.write(yaml_content)
    print("data.yaml successfully created!")

def train_model():
    """
    Trains YOLOv11 small model with custom hyperparameters specifically
    designed for the dataset characteristics found during EDA.
    """
    print("Loading YOLOv11s...")
    model = YOLO('yolo11s.pt') 

    print("Starting training with EDA-optimized custom parameters...")
    model.train(
        data='/kaggle/working/data.yaml', 
        epochs=35,                 
        imgsz=1280,                # High resolution prevents pixel loss for tiny objects
        batch=8,                   
        max_det=500,               # Max objects observed was 359
        
        # Hardening Strategy for Blur and IR
        hsv_s=0.2,                 # Low saturation forces shape learning
        mosaic=1.0,                
        mixup=0.15,                
        copy_paste=0.1,            
        
        project='/kaggle/working/', 
        name='leonardo_airborne_v1', 
        patience=8,                
        optimizer='auto',
        device='0'                 
    )

def generate_submission():
    """
    Runs inference on the test set and outputs submission.csv.
    """
    best_model_path = '/kaggle/working/leonardo_airborne_v1/weights/best.pt'
    if not os.path.exists(best_model_path):
        print(f"Model weights not found at {best_model_path}. Please train first.")
        return

    infer_model = YOLO(best_model_path)
    test_images = sorted(glob(os.path.join(TEST_IMG_DIR, '*.png')))
    
    if not test_images:
        print(f"No test images found in {TEST_IMG_DIR}")
        return

    ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}
    submission_data = []

    print("Running inference on hidden Kaggle test dataset...")
    for img_path in tqdm(test_images, desc="Predicting Test Data"):
        img_name = os.path.basename(img_path).split('.')[0]
        
        results = infer_model.predict(img_path, imgsz=1280, conf=0.03, iou=0.45, max_det=500, verbose=False)[0]
        
        if len(results.boxes) == 0:
            submission_data.append([img_name, "None 1 -1 -1 -1 -1"])
        else:
            boxes_rel = results.boxes.xyxyn.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            pred_strings = []
            for box, conf, cls in zip(boxes_rel, confs, classes):
                class_name = ID_TO_CLASS[cls]
                x_min, y_min, x_max, y_max = box
                
                # Clip bounds exactly to [0.0, 1.0]
                x_min, y_min = np.clip([x_min, y_min], 0.0, 1.0)
                x_max, y_max = np.clip([x_max, y_max], 0.0, 1.0)
                
                pred_str = f"{class_name} {conf:.5f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}"
                pred_strings.append(pred_str)
                
            submission_data.append([img_name, " ".join(pred_strings)])

    sub_df = pd.DataFrame(submission_data, columns=['ImageId', 'PredictionString'])
    sub_df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Success! submission.csv is ready.")

if __name__ == "__main__":
    # convert_dataset()
    # create_yaml()
    # train_model()
    # generate_submission()
    print("Script loaded. Uncomment the __main__ block to run sequentially on Kaggle.")
