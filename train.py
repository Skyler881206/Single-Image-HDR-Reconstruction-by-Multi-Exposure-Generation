import sys
sys.path.append("/work/u8083200/Thesis/SOTA/MEGHDR")

from module import arch as module
from module.loss import loss_fn
import dataset
import config
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import matplotlib.pyplot as plt
import random
import utils

if __name__ == "__main__":
    
    print("Load Config")
    data_root = config.DATA_ROOT
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCH
    aug = config.AUG
    
    weight_name = config.WEIGHT_NAME
    result_save_path = os.path.join(config.RESULT_SAVE_PATH)
    weight_save_path = os.path.join(config.WEIGHT_SAVE_PATH) 
    learning_rate = config.LEARNING_RATE
    device = config.DEVICE
    
    result_root = os.path.join(result_save_path, weight_name)
    
    print("\nWEIGHT_NAME: {}".format(weight_name))
    print("AUG: {}".format(aug))
    
    # ----- Load Data -----
    print("Load Data...")
    hdr_path = []
    ldr_path = []
    
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
        
    if not os.path.exists(result_root):
        os.mkdir(result_root)
        os.mkdir(os.path.join(result_root, "train"))
        os.mkdir(os.path.join(result_root, "val"))
    
    if not os.path.exists(weight_save_path):
        os.mkdir(weight_save_path)
    
    for root, dirs, files in os.walk(data_root):
        files.sort()
        for file in files:
            if(".hdr" in file or ".tif" in file):
                hdr_path.append(os.path.join(root, file))
                continue
    
    # Validation data
    val_path = []
    val_root = "/work/u8083200/Thesis/datasets/HDR-Real"
    for root, dirs, files in os.walk(val_root):
        files.sort()
        for file in files:
            if("gt.hdr" in file):
                val_path.append(os.path.join(root, file))
                continue
    
    random.seed(2454)
    Train_HDR = hdr_path[:int(len(hdr_path) * 0.9)]
    Val_HDR = hdr_path[int(len(hdr_path) * 0.9):]
    
    train_dataloader = DataLoader(dataset.dataset(Train_HDR, stage=4, image_size=512, aug=aug), shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(dataset.dataset(Val_HDR, stage=4, image_size=512, aug=aug), shuffle=True, batch_size=batch_size)
    
    print("Set Model")
    model = module.EDNet()
    model.to(device)
    
    config.set_random_seed(2454)
    
    if os.path.isfile(os.path.join(os.path.join(weight_save_path, weight_name), "best.pth")):
        max_epoch = 0 # Setting LR
        for root, dirs, files in os.walk(os.path.join(weight_save_path, weight_name)):
            if("best.pth" not in files):
                break
            files.remove ("best.pth")
            for file in files:
                if int(file[:-4]) > max_epoch:
                    max_epoch = int(file[:-4])
                    
        model.load_state_dict(torch.load(os.path.join(os.path.join(weight_save_path, weight_name), str(max_epoch) + ".pth")))
        weight_name = weight_name + "_conti"
        
        result_root = os.path.join(result_save_path, weight_name)
        
        if not os.path.exists(result_root):
            os.mkdir(result_root)
            os.mkdir(os.path.join(result_root, "train"))
            os.mkdir(os.path.join(result_root, "val"))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=5e-1, patience=5, verbose=True)
    
    writer = SummaryWriter("runs/" + weight_name + "_log_" + time.ctime(time.time()), 
                           comment=weight_name)
    
    loss_dir = {}
    
    loss_weight = config.loss_weight
    fig, ax = plt.subplots()
    bars = ax.bar(*zip(*loss_weight.items()))
    ax.bar_label(bars)
    writer.add_figure("Loss Weight", fig)
    
    if not os.path.exists(os.path.join(weight_save_path, weight_name)):
        os.mkdir(os.path.join(weight_save_path, weight_name))
    
    
    print("Model Name: {}".format(weight_name))
    # Start Training Section
    train_iteration = 0
    val_iteration = 0
    test_iteration = 0
    min_loss = 9999
    for epoch in range(epochs):
        tqdm_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1} / {epochs}",
                        total=int(len(train_dataloader)))
        
        # Init Loss value, epoch sample count
        loss_dir = {key: 0 for key in loss_dir}
        
        epoch_sample = 0 
        # Training
        for batch_idx, imgs in enumerate(tqdm_bar):
            model.train()
            source = imgs["source"]
            
            source = utils.transfer_to_cuda_and_float32(source)
            
            optimizer.zero_grad()
            
            # Model Run
            up_encode_feature, up_output = model(source, "train_up")
            under_encode_feature, under_output = model(source, "train_under")
            
            predict_dict = {
                "encode_feature": {"x_1": up_encode_feature, "x_2": under_encode_feature},
                "predict": {"x_1": under_output, "x_2": up_output},
                "gt": {"x_1": source["x_1"], "x_2": source["x_2"]},
                "exposure_time": {"t_1": source["t_1"], "t_2": source["t_2"]}
                }

            loss, loss_dict = loss_fn(predict_dict,
                                      loss_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_sample += 1
            
            for key, value in loss_dict.items():
                if batch_idx == 0:
                    loss_dir[key] = loss_dict[key].item()
                    continue
                
                loss_dir[key] += loss_dict[key].item()
            
            if (train_iteration % 999 == 0):
                writer.add_image("Train_Source/Source_x1", dataset.eval_image(source["x_1"]), train_iteration)
                writer.add_image("Train_Source/Source_x2", dataset.eval_image(source["x_2"]), train_iteration)
                writer.add_image("Train_Predict/x_1", dataset.eval_image(predict_dict["predict"]["x_1"]), train_iteration)
                writer.add_image("Train_Predict/x_2", dataset.eval_image(predict_dict["predict"]["x_2"]), train_iteration)
                writer.flush()
            train_iteration += 1
            
            del loss_dict, loss, up_encode_feature, up_output, under_encode_feature, under_output, predict_dict
            
            loss_saving = {key: value / epoch_sample for key, value in loss_dir.items()}
            tqdm_bar.set_postfix(loss_saving)
        
        for key, value in loss_dir.items():
            writer.add_scalar("Train/" + key, loss_dir[key] / epoch_sample, epoch)
            

        freq = 1
        if (epoch + 1) % freq == 0:
            tqdm_bar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1} / {epochs}',
                            total=int(len(val_dataloader)))
            
            loss_dir = {key: 0 for key in loss_dir}
            
            epoch_sample = 0 
            
            for batch_idx, imgs in enumerate(tqdm_bar):
                model.eval()
                
                with torch.no_grad():
                    source = imgs["source"]
                    
                    source = utils.transfer_to_cuda_and_float32(source)
                    
                    optimizer.zero_grad()
                    
                    # Model Run
                    up_encode_feature, up_output = model(source, "train_up")
                    under_encode_feature, under_output = model(source, "train_under")
                    
                    predict_dict = {
                        "encode_feature": {"x_1": up_encode_feature, "x_2": under_encode_feature},
                        "predict": {"x_1": under_output, "x_2": up_output},
                        "gt": {"x_1": source["x_1"], "x_2": source["x_2"]},
                        "exposure_time": {"t_1": source["t_1"], "t_2": source["t_2"]}
                        }

                    loss, loss_dict = loss_fn(predict_dict,
                                              loss_weight)
                    
                    epoch_sample += 1
                    
                    for key, value in loss_dict.items():
                        if batch_idx == 0:
                            loss_dir[key] = loss_dict[key].item()
                            continue
                
                        loss_dir[key] += loss_dict[key].item()
                        
                    if (val_iteration % 59 == 0):
                        writer.add_image("val_Source/Source_x1", dataset.eval_image(source["x_1"]), val_iteration)
                        writer.add_image("val_Source/Source_x2", dataset.eval_image(source["x_2"]), val_iteration)
                        writer.add_image("val_Predict/x_1", dataset.eval_image(predict_dict["predict"]["x_1"]), val_iteration)
                        writer.add_image("val_Predict/x_2", dataset.eval_image(predict_dict["predict"]["x_2"]), val_iteration)
                        writer.flush()
                    val_iteration += 1
        
                    del loss_dict, loss, up_encode_feature, up_output, under_encode_feature, under_output, predict_dict
        
                    loss_saving = {key: value / epoch_sample for key, value in loss_dir.items()}
                    tqdm_bar.set_postfix(loss_saving)
                    
            scheduler.step(loss_dir["loss"] / epoch_sample)
            for key, value in loss_dir.items():
                writer.add_scalar("Val/" + key, loss_dir[key] / epoch_sample, epoch)
                
        if((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), os.path.join(os.path.join(weight_save_path, weight_name), str(epoch+1)+'.pth'))
        if(loss_dir["loss"] / epoch_sample < min_loss):
            min_loss = loss_dir["loss"] / epoch_sample
            torch.save(model.state_dict(), os.path.join(os.path.join(weight_save_path, weight_name), 'best.pth'))
            
    print("Finish Training :)")