import json
import os
import pydicom
import cv2
import shutil
from sklearn.model_selection import train_test_split


# 1. Load the adjudicated annotations
with open('../pneumonia-challenge-dataset-mappings_2018.json') as f:
    data = json.load(f)

def build_dataset(base_source_path):
    # Prepare the list to track IDs and their targets
    valid_entries = []
    
    for entry in data:
        uid = entry['StudyInstanceUID']
        # Use subset_init_label based on your provided key
        label = entry.get('subset_init_label')
        
        if label == 1:
            valid_entries.append((uid, 'pneumonia'))
        elif label == 2:
            valid_entries.append((uid, 'normal'))
        # label 0 is ignored for better accuracy
            
    # 2. Split into Train (80%), Val (10%), and Test (10%)
    train_data, temp_data = train_test_split(valid_entries, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    # 3. Process and Move Files
    for split_name, entries in splits.items():
        for uid, category in entries:
            # Find the .dcm file in the deep nested folders
            patient_path = os.path.join(base_source_path, uid)
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.endswith(".dcm"):
                        # Read and Standardize (224x224x3)
                        ds = pydicom.dcmread(os.path.join(root, file))
                        img = ds.pixel_array
                        img_resized = cv2.resize(img, (224, 224))
                        
                        # Convert grayscale to RGB depth 3 for VGG16
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                        
                        # Save to organized directory
                        target_dir = f"data/{split_name}/{category}"
                        os.makedirs(target_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(target_dir, f"{uid}.png"), img_rgb)
# At the very bottom of your data_pipeline.py
    

if __name__ == "__main__":
    build_dataset('../Pneumonia_Dataset')