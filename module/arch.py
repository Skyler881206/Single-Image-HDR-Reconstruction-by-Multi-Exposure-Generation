import torch
import torch.nn as nn

def normalized_tanh(x, inplace: bool = False):
    return 0.5 * x.tanh() + 0.5

class NormalizedTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(NormalizedTanh, self).__init__()

    def forward(self, x):
        return normalized_tanh(x)

class base_last_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(base_last_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.block(x)

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        return self.block(x)

class downsample_block(nn.Module):
    def __init__(self, in_channels):
        super(downsample_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2),
        )
        
    def forward(self, x):
        return self.block(x)

class upsample_block(nn.Module):
    def __init__(self, in_channels):
        super(upsample_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        return self.block(x)

class Unet_encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels, max_feature):
        super(Unet_encoder, self).__init__()
        self.encoder_1 = encoder_block(in_channels, min(encoder_channels, max_feature))
        self.encoder_2 = encoder_block(min(encoder_channels, max_feature), min(encoder_channels * 2, max_feature))
        self.encoder_3 = encoder_block(min(encoder_channels * 2, max_feature), min(encoder_channels * 4, max_feature))
        self.encoder_4 = encoder_block(min(encoder_channels * 4, max_feature), min(encoder_channels * 8, max_feature))
        self.encoder_5 = encoder_block(min(encoder_channels * 8, max_feature), min(encoder_channels * 16, max_feature))
        self.encoder_6 = encoder_block(min(encoder_channels * 16, max_feature), min(encoder_channels * 32, max_feature))
        self.encoder_7 = encoder_block(min(encoder_channels * 32, max_feature), min(encoder_channels * 64, max_feature))
        
        self.downsample_1 = downsample_block(min(encoder_channels, max_feature))
        self.downsample_2 = downsample_block(min(encoder_channels * 2, max_feature))
        self.downsample_3 = downsample_block(min(encoder_channels * 4, max_feature))
        self.downsample_4 = downsample_block(min(encoder_channels * 8, max_feature))
        self.downsample_5 = downsample_block(min(encoder_channels * 16, max_feature))
        self.downsample_6 = downsample_block(min(encoder_channels * 32, max_feature))
        

    
    def forward(self, x):
        feature_1 = self.encoder_1(x) # 1/1
        feature_2 = self.encoder_2(self.downsample_1(feature_1)) # 1/2
        feature_3 = self.encoder_3(self.downsample_2(feature_2)) # 1/4
        feature_4 = self.encoder_4(self.downsample_3(feature_3)) # 1/8
        feature_5 = self.encoder_5(self.downsample_4(feature_4)) # 1/16
        feature_6 = self.encoder_6(self.downsample_5(feature_5)) # 1/32
        feature_7 = self.encoder_7(self.downsample_6(feature_6)) # 1/64
        
        return (feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7)
    

class Unet_decoder(nn.Module):
    def __init__(self, decoder_channels, max_feature, res_type = "concat"):
        super(Unet_decoder, self).__init__()
        
        self.res_type = res_type
        
        if res_type == "concat":
            self.decoder_7 = decoder_block(min(decoder_channels * 64, max_feature),
                                        min(decoder_channels * 32, max_feature))
            
            self.decoder_6 = decoder_block(min(decoder_channels * 32, max_feature) + min(decoder_channels * 32, max_feature),
                                        min(decoder_channels * 16, max_feature))
            
            self.decoder_5 = decoder_block(min(decoder_channels * 16, max_feature) + min(decoder_channels * 16, max_feature),
                                        min(decoder_channels * 8, max_feature))
            
            self.decoder_4 = decoder_block(min(decoder_channels * 8, max_feature) + min(decoder_channels * 8, max_feature),
                                        min(decoder_channels * 4, max_feature))  
            
            self.decoder_3 = decoder_block(min(decoder_channels * 4, max_feature) + min(decoder_channels * 4, max_feature),
                                        min(decoder_channels * 2, max_feature))
            
            self.decoder_2 = decoder_block(min(decoder_channels * 2, max_feature) + min(decoder_channels * 2, max_feature),
                                        min(decoder_channels * 1, max_feature))
            
            self.decoder_1 = decoder_block(min(decoder_channels * 1, max_feature) + min(decoder_channels, max_feature),
                                        min(decoder_channels, max_feature))
        
        if res_type == "add":
            self.decoder_7 = decoder_block(min(decoder_channels * 64, max_feature),
                                        min(decoder_channels * 32, max_feature))
            
            self.decoder_6 = decoder_block(min(decoder_channels * 32, max_feature),
                                        min(decoder_channels * 16, max_feature))
            
            self.decoder_5 = decoder_block(min(decoder_channels * 16, max_feature),
                                        min(decoder_channels * 8, max_feature))
            
            self.decoder_4 = decoder_block(min(decoder_channels * 8, max_feature),
                                        min(decoder_channels * 4, max_feature))  
            
            self.decoder_3 = decoder_block(min(decoder_channels * 4, max_feature),
                                        min(decoder_channels * 2, max_feature))
            
            self.decoder_2 = decoder_block(min(decoder_channels * 2, max_feature),
                                        min(decoder_channels * 1, max_feature))
            
            self.decoder_1 = decoder_block(min(decoder_channels, max_feature),
                                        min(decoder_channels, max_feature))
        
        self.up_7 = upsample_block(min(decoder_channels * 32, max_feature))
        self.up_6 = upsample_block(min(decoder_channels * 16, max_feature))
        self.up_5 = upsample_block(min(decoder_channels * 8, max_feature))
        self.up_4 = upsample_block(min(decoder_channels * 4, max_feature))
        self.up_3 = upsample_block(min(decoder_channels * 2, max_feature))
        self.up_2 = upsample_block(min(decoder_channels * 1, max_feature))
        
    def forward(self, x):
        
        if self.res_type == "concat":
            feature = self.decoder_7(x[-1]) # 1/64
            feature = self.decoder_6(torch.cat([self.up_7(feature), x[-2]], dim=1)) # 1/32
            feature = self.decoder_5(torch.cat([self.up_6(feature), x[-3]], dim=1)) # 1/16
            feature = self.decoder_4(torch.cat([self.up_5(feature), x[-4]], dim=1)) # 1/8
            feature = self.decoder_3(torch.cat([self.up_4(feature), x[-5]], dim=1)) # 1/4
            feature = self.decoder_2(torch.cat([self.up_3(feature), x[-6]], dim=1)) # 1/2
            feature = self.decoder_1(torch.cat([self.up_2(feature), x[-7]], dim=1)) # 1/1
            
        elif self.res_type == "add":
            feature = self.decoder_7(x[-1])
            feature = self.decoder_6(self.up_7(feature) + x[-2])
            feature = self.decoder_5(self.up_6(feature) + x[-3])
            feature = self.decoder_4(self.up_5(feature) + x[-4])
            feature = self.decoder_3(self.up_4(feature) + x[-5])
            feature = self.decoder_2(self.up_3(feature) + x[-6])
            feature = self.decoder_1(self.up_2(feature) + x[-7])
            
        return feature

class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.HDREncNet = nn.Sequential(
            Unet_encoder(3, 16, 256),
            Unet_decoder(16, 256, res_type = "add"),
            base_last_layer(16, 3),
            nn.Tanh(),
        )
        
        self.UpexosureNet = nn.Sequential(
            Unet_encoder(3, 32, 512),
            Unet_decoder(32, 512, res_type = "concat"),
            base_last_layer(32, 3),
            NormalizedTanh(),
        )
        
        self.UnderexposureNet = nn.Sequential(
            Unet_encoder(3, 32, 512),
            Unet_decoder(32, 512, res_type = "concat"),
            base_last_layer(32, 3),
            NormalizedTanh(),
        )
        
    def forward(self, x, train_section = None):
        if train_section == "train_up":
            Encode_feature = self.HDREncNet(x["x_1"] * self.over_under_mask(x["x_1"]))
            Encode_feature = (Encode_feature + x["x_1"] + 1.0) / 3.0
            Up_feature = self.UpexosureNet(Encode_feature * (x["t_2"] / x["t_1"]).view(-1, 1, 1, 1))
            return (Encode_feature, Up_feature)
        
        elif train_section == "train_under":
            Encode_feature = self.HDREncNet(x["x_2"] * self.over_under_mask(x["x_2"]))
            Encode_feature = (Encode_feature + x["x_2"] + 1.0) / 3.0
            Under_feature = self.UnderexposureNet(Encode_feature * (x["t_1"] / x["t_2"]).view(-1, 1, 1, 1))
            return (Encode_feature, Under_feature)
        
        elif train_section == "val":
            Encode_feature = self.HDREncNet(x * self.over_under_mask(x))
            Encode_feature = (Encode_feature + x + 1.0) / 3.0
            Up_feature = self.UpexosureNet(Encode_feature * (x["t_2"] / x["t_1"]).view(-1, 1, 1, 1))
            Under_feature = self.UnderexposureNet(Encode_feature * (x["t_1"] / x["t_2"]).view(-1, 1, 1, 1))
            return (Encode_feature, Up_feature, Under_feature)
        
        else:
            raise("train_section must be one of ['train_up', 'train_under', 'val']")
        
        
    @staticmethod
    def over_under_mask(x, gamma = 0.05):
        x = torch.mean(x, dim=1, keepdim=True)
        over_mask = 1 - (torch.max(torch.zeros_like(x), (1 - gamma) - x) / (1 - gamma))
        under_mask  = 1 - (torch.max(torch.zeros_like(x), (x - gamma)) / (1 - gamma))
        return torch.max(over_mask, under_mask)
        
if __name__ == "__main__":
    model = EDNet().to("cuda")
    
    test_tensor = torch.randn(4, 3, 512, 512).to("cuda")
    test_tensor = torch.clamp(test_tensor, 0, 1)
    
    output = model(test_tensor, "val")
    print(output.shape)