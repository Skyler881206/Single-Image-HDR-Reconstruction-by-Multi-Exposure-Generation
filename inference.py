from module import arch as module
from utils import merge_hdr
from tqdm import tqdm
import torch
import os
import cv2
import numpy as np
import config
import dataset

def copy_folder_structure(source_folder, destination_folder):
    # Get all the items (folders and files) in the source folder
    items = os.listdir(source_folder)

    for item in items:
        item_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isdir(item_path):
            # If the item is a directory, create the corresponding directory in the destination folder
            os.makedirs(destination_path)
            
            # Recursively copy the subfolder's structure
            copy_folder_structure(item_path, destination_path)

if __name__ == "__main__":
    
    dataset_root = config.TEST_DATA_ROOT
    model_name = config.WEIGHT_NAME
    
    datasets = {"HDR_Eye": "HDR-Eye",
               "HDR_Real": "HDR_Real",
               "HDR_Synth": "HDR-Synth",
               "RAISE": "RAISE"}
    
    # copy_folder_structure(dataset_root, config.RESULT_SAVE_PATH)
    
    file_root = []
    save_root = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if "input.jpg" in file:
                file_root.append(os.path.join(root))
                save_root.append(os.path.join(root.replace(dataset_root, config.VAL_RESULT_SAVE_PATH)))
                
    model = module.EDNet(config.EXPOSURE_TIME).to("cuda")
    model.load_state_dict(torch.load(os.path.join(config.WEIGHT_SAVE_PATH, config.WEIGHT_NAME, "best.pth")))  
    model.eval()
    
    print("Model Name: " + model_name)
    
    tqdm_bar = tqdm(file_root, desc="Save iter")
    for idx, file_root in enumerate(tqdm_bar):
        input_ldr_path = os.path.join(file_root, "input.jpg")
        input_ldr = cv2.imread(input_ldr_path).astype(np.float32) / 255.0
        input_ldr = torch.from_numpy(np.transpose(input_ldr, (2, 0, 1)))
        input_ldr = torch.unsqueeze(input_ldr, 0).to("cuda")
        
        prediction = model(input_ldr, "val")
        prediction = {"images": [input_ldr, prediction[1], prediction[2]],
                      "times": [1.0,
                                config.EXPOSURE_TIME["t_2"] / config.EXPOSURE_TIME["t_1"],
                                config.EXPOSURE_TIME["t_1"] / config.EXPOSURE_TIME["t_2"]]}
        
        prediction = merge_hdr(prediction)
        
        dataset.save_hdr_image(prediction, os.path.join(save_root[idx], model_name + ".hdr"))

