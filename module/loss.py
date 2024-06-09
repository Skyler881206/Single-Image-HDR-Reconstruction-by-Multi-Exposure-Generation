import torch
import lpips

l1_loss = torch.nn.L1Loss()
p_loss = lpips.LPIPS(net='vgg').to(device="cuda")
def loss_fn(predict, loss_weight):
    def transformation_loss(predict_1, predict_2, exposure_time):
        return l1_loss(torch.log(predict_1 * (exposure_time[1] / exposure_time[0]).view(-1, 1, 1, 1) + 1e-6), torch.log(predict_2 + 1e-6))
    
    def representation_loss(encode_feature, exposure_time):
        return transformation_loss(encode_feature["x_1"], encode_feature["x_2"], [exposure_time["t_1"], exposure_time["t_2"]]) + \
               transformation_loss(encode_feature["x_2"], encode_feature["x_1"], [exposure_time["t_2"], exposure_time["t_1"]])
    
    def reconstruction_loss(predict, gt):
        return l1_loss(predict["x_1"], gt["x_1"]) + l1_loss(predict["x_2"], gt["x_2"])
    
    def perceptual_loss(predict, gt):
        return torch.mean(p_loss(predict["x_1"], gt["x_1"])) + torch.mean(p_loss(predict["x_2"], gt["x_2"]))
    
    def tv_loss(predict):
        return total_variation(predict["x_1"]) + total_variation(predict["x_2"])  
    
    def total_variation(predict):
        batch_size, channels, target_h, target_w= predict.shape
        h_tv = torch.abs(predict[:,:,1:,:] - predict[:,:,:-1,:]).sum()
        w_tv = torch.abs(predict[:,:,:,1:] - predict[:,:,:,:-1]).sum()

        return (h_tv + w_tv) / (batch_size * channels * target_h * target_w)
    
    # predict : {"encode_feature": {x_1: torch.Tensor, x_2: torch.Tensor},
    #            "predict": {x_1: torch.Tensor, x_2: torch.Tensor}},
    #            "gt" : {"x_1": torch.Tensor, "x_2": torch.Tensor},
    #            "exposure_time": {"t_1": float, "t_2": float}}
    
    loss_dict = {}
    for key, val in loss_weight.items():
        if key == "representation_loss":
            loss_dict[key] = representation_loss(predict["encode_feature"], predict["exposure_time"])
        elif key == "reconstruction_loss":
            loss_dict[key] = reconstruction_loss(predict["predict"], predict["gt"])
        elif key == "perceptual_loss":
            loss_dict[key] = perceptual_loss(predict["predict"], predict["gt"])
        elif key == "tv_loss":
            loss_dict[key] = tv_loss(predict["predict"])
        
    for idx, (key, val) in enumerate(loss_dict.items()):
        if idx == 0:
            loss = val * loss_weight[key]
            continue
        loss += val * loss_weight[key]
    loss_dict["loss"] = loss
    return loss, loss_dict