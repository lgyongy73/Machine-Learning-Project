import json
import os
import pydicom #for reading the. dcm files
import cv2 #for data processing (resizing & greyscale to RGB)
import shutil
from sklearn.model_selection import train_test_split

#the IDs of the pictures based on patient numbers, 1 for diseased 2 for healthy and 0 for unknown (we handle 0 differently)
with open('../pneumonia-challenge-dataset-mappings_2018.json') as f:
    data = json.load(f)

def build_dataset(base_source_path):
    valid_entries = []
    
    for entry in data:
        uid = entry['StudyInstanceUID'] #where the patients are in the file
        label = entry.get('subset_init_label') #their ID
        
        if label == 1:
            valid_entries.append((uid, 'pneumonia'))
        elif label == 2:
            valid_entries.append((uid, 'normal'))
    # we chose to ignore and not catgorise pictures with 0 ID, since it is unkown, we could ruin the accuracy
    # of the method, we also can use it to test, since we dont know the correct answer
                
    # split into the 3 folders, training data, validation data, and test data
    train_data, temp_data = train_test_split(valid_entries, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    # processing (resizing and grayscale to RGB) and splitting the data into their respective folders
    for split_name, entries in splits.items():

        for uid, category in entries:
            patient_path = os.path.join(base_source_path, uid)

            for root, dirs, files in os.walk(patient_path):
            # os.walk goes into the subfolders because the actual dcm file is in a folder
            # in another folder, in the main folder ("Pneumonia_Dataset")
                for file in files:
                    if file.endswith(".dcm"):
                        ds = pydicom.dcmread(os.path.join(root, file))
                        img = ds.pixel_array
                        img_resized = cv2.resize(img, (224, 224))
                        #reads the dcm file using the built in python pydicom library (made for reading dcm files)
                        
                        #the VGG16 model need RGB as that is what it was trained on, so we convert
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                        
                        target_dir = f"data/{split_name}/{category}"
                        #the final path after categorisation
                        os.makedirs(target_dir, exist_ok=True)
                        #doesnt make new one if it already exists
                        cv2.imwrite(os.path.join(target_dir, f"{uid}.png"), img_rgb)
                        #saved as png using the ID for easy recognition
    

if __name__ == "__main__":
    #only runs when explicitly called

    build_dataset('../Pneumonia_Dataset')
    #this is why i wrote in the README to put the two into a main master folder
    #it steps out of the git folder and looks for the dataset