# It's two stage training module for single hdr image reconstruction
import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
# from skimage.feature import canny
import random
ROOT_PATH = "/work/u8083200/Thesis/Test_arch/Easy_SingleHDR"

np.seterr(all="ignore")

def _get_crf_list():
    with open(os.path.join(ROOT_PATH, 'dorfCurves.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
    crf_list = np.float32([ele.split() for ele in crf_list])
    np.random.RandomState(730).shuffle(crf_list)
    
    train_crf_list = crf_list[:-10]
    test_crf_list = crf_list[-10:]
    return train_crf_list, test_crf_list

train_crf_list, test_crf_list= _get_crf_list()
class dataset(Dataset):
    def __init__(self, hdr_path=None, ldr_path=None, crf_list=train_crf_list,
                 aug=True, aug_ev=4, image_size=512, stage=1, sigma=2.0):
        self.hdr_path = hdr_path
        self.ldr_path = ldr_path
        self.crf_list = crf_list
        self.aug = aug
        self.aug_ev = aug_ev
        self.image_size = image_size
        self.stage = stage
        self.sigma = sigma
        '''
        Stage
        1: Transfer Domain Stage
        2: Inpaint Stage
        3: Refinement Stage
        4: Transfer HDR Domain Stage
        5: Refinement Stage (Using Real LDR image)
        '''

    def __len__(self):
        return len(self.hdr_path)

    def __getitem__(self, index):
        RandomCrop_transform = transforms.RandomCrop(self.image_size)
        if ".tif" in self.hdr_path[index]:
            hdr_image = cv2.imread(self.hdr_path[index], flags=cv2.IMREAD_UNCHANGED) # Read HDR image / Model Target
        else:
            hdr_image = cv2.imread(self.hdr_path[index], flags=cv2.IMREAD_ANYDEPTH) # Read HDR image / Model Target
        
        if hdr_image.min() < 0.0:
            hdr_image = hdr_image + abs(hdr_image.min())
            
        hdr_image_correction = self.lights_metering(hdr_image) # Change the image to same metering level
        hdr_image = hdr_image_correction.copy()
        
        if self.stage == 5: # For refinement
            ldr_image = cv2.imread(self.ldr_path[index]).astype(np.float32) / 255.0 # Read HDR image
        
        if self.aug == True:
            flip_ = random.random()
            rot_ = random.random()
            if flip_ > 0.5: # Rotate
                hdr_image_correction = np.flip(hdr_image_correction, round(random.random()))
                if self.stage == 5:
                    ldr_image = np.flip(ldr_image, round(random.random()))
            if rot_ > 0.5:
                hdr_image_correction = np.rot90(hdr_image_correction, round(random.random() * 3), (0, 1))
                if self.stage == 5:
                    ldr_image = np.rot90(ldr_image, round(random.random() * 3), (0, 1))
            ev = random.uniform(-self.aug_ev, self.aug_ev)
            hdr_image = hdr_image_correction * (pow(2, ev)) # Change Exposure
        
        if hdr_image.shape[0] != self.image_size or hdr_image.shape[1] != self.image_size: # If the size is minus than image_size, Resize
            hdr_image = self.resize(hdr_image, self.image_size, self.image_size)

        i, j, h, w = RandomCrop_transform.get_params(torch.zeros(np.transpose(hdr_image, (2, 0, 1)).shape),
                                                    (self.image_size, self.image_size))
            
        # -----Refinement-----
        if(self.stage == 5): # Using Real LDR image refine model
            return{"Target": transforms.functional.crop(torch.from_numpy(np.transpose(hdr_image, (2, 0, 1))), i, j, h, w),
                   "Source": transforms.functional.crop(torch.from_numpy(np.transpose(ldr_image, (2, 0, 1))), i, j, h, w)}
        # -----Refinement-----
        exposure_time = {
            "t_1": 1.0,
            "t_2": np.random.uniform(1, 4)
        }
        pair_hdr_image = hdr_image * exposure_time["t_2"] / exposure_time["t_1"] # Pair HDR image
        
        # -----Hallucination-----
        hdr_clip, _, _ = self.dynamic_clipping(hdr_image)
        pair_hdr_clip, _, _ = self.dynamic_clipping(pair_hdr_image)
        # linear_ldr_8bit = np.round(hdr_clip * 255.0)
        # -----Hallucination-----
        
        # -----Linearization-----
        _get_icrf_list = lambda crf_list: np.array([self.inverse_crf(crf) for crf in crf_list])
        random.shuffle(self.crf_list) # Shuffle crf_list
        icrf_list = _get_icrf_list(self.crf_list)
        
        non_linear_ldr = self.apply_crf(hdr_clip, self.crf_list[0])
        quantized_ldr = np.round(non_linear_ldr * 255.0)
        quantized_ldr_8bit = quantized_ldr.astype(np.uint8)
        quantized_jpeg_encode = cv2.imencode(".jpg", quantized_ldr_8bit)[1]
        quantized_jpeg = cv2.imdecode(quantized_jpeg_encode, cv2.IMREAD_COLOR)
        quantized_jpeg_float = quantized_jpeg.astype(np.float32) / 255.0 # Jpeg Output
        
        pair_non_linear_ldr = self.apply_crf(pair_hdr_clip, self.crf_list[0])
        pair_quantized_ldr = np.round(pair_non_linear_ldr * 255.0)
        pair_quantized_ldr_8bit = pair_quantized_ldr.astype(np.uint8)
        pair_quantized_jpeg_encode = cv2.imencode(".jpg", pair_quantized_ldr_8bit)[1]
        pair_quantized_jpeg = cv2.imdecode(pair_quantized_jpeg_encode, cv2.IMREAD_COLOR)
        pair_quantized_jpeg_float = pair_quantized_jpeg.astype(np.float32) / 255.0 # Jpeg Output
        # ------Edge------
        # hdr_edge, hdr_gray_image = self.image_to_edge(hdr_image, 2.0)
        # hdr_edge, hdr_gray_image = self.image_to_edge(hdr_image, self.sigma)
        # hdr_crf_edge, hdr_crf_gray_image = self.image_to_edge(self.apply_crf(hdr_image, self.crf_list[0]), self.sigma)
        # jpeg_edge, jpeg_gray_image = self.image_to_edge(quantized_jpeg_float, self.sigma)
        # hdr_edge = np.where((hdr_edge + hdr_crf_edge + jpeg_edge) > 0, 1.0, 0.0)
        
        # cv2.imwrite("hdr_edge.jpg", hdr_edge * 255.0)
        # cv2.imwrite("jpeg_edge.jpg", jpeg_edge * 255.0)
        # cv2.imwrite("image.jpg", quantized_jpeg_float * 255.0)
        # cv2.imwrite("hdr_image.jpg", hdr_image_correction * 255.0)
        # ------Edge------        
        if self.stage == 4: # Refinement Stage
            return {
                "source": {
                    "x_1": transforms.functional.crop(torch.from_numpy(np.transpose(quantized_jpeg_float, (2, 0, 1))), i, j, h, w),
                    "x_2": transforms.functional.crop(torch.from_numpy(np.transpose(pair_quantized_jpeg_float, (2, 0, 1))), i, j, h, w),
                    "t_1": torch.tensor(exposure_time["t_1"]),
                    "t_2": torch.tensor(exposure_time["t_2"]) 
                    },
                }
        
        # Tone Mapping Section
        tone_hdr_clip = self.tonemap_operator(hdr_clip, 10) # Tone Mapping HDR_clip
        tone_hdr = self.tonemap_operator(hdr_image, 10) # Tone Mapping HDR
        tone_hdr_max = tone_hdr.max()
        
        correction_hdr = tone_hdr_clip / tone_hdr_max # Normalize the image
        tone_hdr = tone_hdr / tone_hdr_max # Normalize the image
        
        tone_map_hdr_clip = self.tonemap_operator(correction_hdr, 10) # Tone Mapping HDR
        tone_map_hdr = self.tonemap_operator(tone_hdr, 10) # Tone Mapping HDR
        
        # tone_map_hdr, tone_map_hdr_clip = self.histogram_equalization_float(tone_hdr, correction_hdr)
        tone_map_hdr_he, tone_map_hdr_clip_he, he_curve = self.histogram_equalization_float(tone_map_hdr, tone_map_hdr_clip)
        # transform_curve = self.find_curve(non_linear_ldr, tone_map_hdr_clip_he)
        # transform_curve = self.inverse_tonemap_operator(self.inverse_tonemap_operator(he_curve) + tone_hdr_max)
        transform_curve = self.apply_crf(self.tonemap_operator(self.tonemap_operator(icrf_list[0], 10) / tone_hdr_max, 10), he_curve)
            
        # apply_curve_ = self.apply_crf(tone_map_hdr_clip, he_curve)
        # apply_curve = self.apply_crf(non_linear_ldr, transform_curve)
        
        # app = self.apply_crf(non_linear_ldr, icrf_list[0])
        # app = self.tonemap_operator(app, 10)
        # app = np.clip((app / tone_hdr_max), 0.0, 1.0)
        # app = self.tonemap_operator(app, 10)
        # app = self.apply_crf(app, he_curve)    
        
        # app = self.apply_crf(non_linear_ldr, transform_curve)
        # dif = (app - tone_map_hdr_clip_he).max() 
        # print(dif)
        # app_curv = np.clip(self.tonemap_operator(icrf_list[0], 10) / tone_hdr_max, 0.0, 1.0)
        # app_curv = self.apply_crf(self.tonemap_operator(app_curv, 10), he_curve)
        # appcv = self.apply_crf(non_linear_ldr, app_curv)
        # hdr_y = np.mean(tone_hdr, axis=2, keepdims=True)
        # soft_mask = np.where(hdr_y - 0.95 > 0, hdr_y - 0.95, 0) / (1.0 - 0.95)
        # soft_mask = np.where(soft_mask > 1.0, 1.0, soft_mask)
        
        # tone_map_hdr = np.where(soft_mask != 1.0, tone_map_hdr, 1.0)
        
        # Cat the image
        cat_image = np.zeros((tone_map_hdr_clip.shape[0], tone_map_hdr_clip.shape[1] * 2, tone_map_hdr_clip.shape[2]))
        cat_image[:, :tone_map_hdr_clip.shape[1], :] = tone_map_hdr_clip
        cat_image[:, tone_map_hdr_clip.shape[1]:, :] = hdr_image
        # cv2.imwrite("tone_map_hdr.jpg", cat_image * 255.0)
        
        # Enhancement Section, To fix the color 
        # img = cv2.imread('/work/u8083200/Thesis/datasets/SingleHDR_training_data/HDR-Real/LDR_in/02092.jpg')
        
        # T_c = self.ImageEnhance(quantized_jpeg) # Enhance the image display
        
        # if True in np.isnan(T_c):
        #     # logging.warning("\nNan in T_c:{}".format(self.hdr_path[index]))
        #     T_c = np.nan_to_num(T_c, copy=False, nan=0.0)
            
        # T_c = self.lights_metering(quantized_jpeg_float) # Change The image to same metering level
        if self.stage == 1: # Transfer Domain Stage
            return {"target": transforms.functional.crop(torch.from_numpy(np.transpose(tone_map_hdr_clip_he, (2, 0, 1))), i, j, h, w),
                    "deq_target": transforms.functional.crop(torch.from_numpy(np.transpose(non_linear_ldr, (2, 0, 1))), i, j, h, w),
                    "source": transforms.functional.crop(torch.from_numpy(np.transpose(quantized_jpeg_float, (2, 0, 1))), i, j, h, w),
                    # "source_tc": transforms.functional.crop(torch.from_numpy(np.transpose(T_c, (2, 0, 1))), i, j, h, w),
                    "he_curve": torch.from_numpy(transform_curve)}
        
        if self.stage == 2: # Inpaint Stage
            return {"ori_image": transforms.functional.crop(torch.from_numpy(np.transpose(quantized_jpeg_float, (2, 0, 1))), i, j, h, w),
                    "target": transforms.functional.crop(torch.from_numpy(np.transpose(tone_map_hdr_he, (2, 0, 1))), i, j, h, w),
                    "source": transforms.functional.crop(torch.from_numpy(np.transpose(tone_map_hdr_clip_he, (2, 0, 1))), i, j, h, w)}
            
        if self.stage == 3: # Transfer hdr domain Stage
            return {"target": transforms.functional.crop(torch.from_numpy(np.transpose(hdr_image_correction, (2, 0, 1))), i, j, h, w),
                    "source": transforms.functional.crop(torch.from_numpy(np.transpose(tone_map_hdr, (2, 0, 1))), i, j, h, w)}
            
        # -----Linearization-----
        return None
    
    @staticmethod
    def image_to_edge(image, sigma):
        gray_image = np.mean(image, axis=2) 
        edge = canny(gray_image, sigma=sigma)
        gray_image = np.expand_dims(gray_image, axis=2)
        
        edge = np.expand_dims(edge, axis=2)
        edge = np.where(edge == True, 1.0, 0.0)
        return edge, gray_image      
    
    @staticmethod
    def dynamic_clipping(image) -> np.ndarray: # HDR -> Clipping to Linear LDR
        over_exposure_mask = np.where(np.mean(image, axis=2, keepdims=True) > 0.95, 1, 0)
        under_exposure_mask = np.where(np.mean(image, axis=2, keepdims=True) < 0.05, 1, 0)
        image = np.where(image <= 1, image, 1)
        return image, over_exposure_mask, under_exposure_mask
    
    @staticmethod
    def resize(image, h, w) -> np.ndarray:
        image = cv2.resize(image, (h, w), cv2.INTER_AREA)
        return image
    
    @staticmethod
    def apply_crf(x, crf): # Return non-linear LDR
        # return interp1d([length for length in range(len(crf))], crf)(x * (len(crf)-1))
        return interp1d(np.linspace(0.0, 1.0, num=len(crf)), crf,
                        fill_value='extrapolate', bounds_error=False)(x)
    
    @staticmethod
    def inverse_crf(crf): # Get iCRF use inverse function method
        _crf = crf.copy()
        # _crf[0] = 0 # Minimum of CRF should be zero.
        # _crf[-1] = 1 # Maximum of CRF should be one.
        
        return interp1d(_crf, np.linspace(0.0, 1.0, num=len(_crf)))(np.linspace(0.0, 1.0, num=len(_crf)))
    
    @staticmethod
    def ImageEnhance(image): # Input shape (H, W, 3), uint8 format
        Ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y_image = Ycbcr_image[:, :, 0]
        bf_Y = cv2.bilateralFilter(Y_image, 7, 13, 13) / 255.0
        
        y_mean = np.mean(Y_image)
        Y_a = 0.5 + np.power(((y_mean / 255.0) - bf_Y), 3)
        
        # Nonlinear Modification to Histogram
        hist, bins = np.histogram(Y_image, 256, [0, 256])
        hist = hist / np.sum(hist) # Normalize
        
        weight = np.abs(bins[:-1] - 127) / 255.0 + 0.7
        
        F_x = np.zeros(Y_image.shape)
        
        hist_h = np.zeros(hist.shape)
        hist_h[int(y_mean) + 1:] = hist[int(y_mean) + 1:]
        
        cdf_l = np.cumsum(hist * weight)
        cdf_h = np.cumsum(hist_h * weight)

        pl = cdf_l[int(np.ceil(y_mean))]
        ph = cdf_h[-1]
        
        F_x = np.where(Y_image < y_mean,
                       Y_a * cdf_l[Y_image] / pl,
                       Y_a + (1 - Y_a) * cdf_h[Y_image] / ph)
        
        # Color Reconstruction
        F_x = (F_x - F_x.min()) / (F_x.max() - F_x.min()) # Normalize
        F_x = np.expand_dims(F_x, axis=2) # [height, width, channel]
        
        # cv2.imwrite("F_x.jpg", F_x * 255.0)
        
        Y_image = np.expand_dims(Y_image, axis=2) # [height, width, channel]
        de_Y_image = np.where(Y_image == 0, 0, 1 / Y_image)
        
        # Y_image = Y_image / 255.0 # Range: [0, 1]
        # image = image / 255.0 # Range: [0, 1]
        # y_mean = y_mean / 255.0 # Range: [0, 1]
        F_x = F_x * 255.0 # Range: [0, 255]
        
        E_c_ = (1/2) * ((image * (F_x * de_Y_image)) + image + F_x - Y_image)
        

        
        E_c = E_c_ + (image - Y_image) * y_mean / (Y_image.max() - Y_image.min())
        E_c = np.where(E_c >= 255.0, 255.0, E_c)
        E_c = np.where(E_c <= 0.0, 0.0, E_c)
        
        E_c = E_c / 255.0 # Range: [0, 1]
        
        # Local Contrast Compensation
        # bf_E_c = cv2.bilateralFilter((E_c * 255.0).astype(np.uint8), 5, 2.0, 0.1)
        bf_E_c = cv2.bilateralFilter((E_c).astype(np.uint8), 7, 13, 13) / 255
        bf_E_c = bf_E_c / 255.0
        
        # if 0 or 1.0 in E_c:
        #     E_c = np.where(E_c == 0, 1e-5, E_c)
        #     E_c = np.where(E_c == 1.0, 1 - 1e-5, E_c)
        
        de_E_c = np.where(E_c == 0, 0, 1 / E_c)
        de_1_E_c = np.where(1 - E_c == 0, 0, 1 - E_c)
        E_c_h = np.power(E_c, bf_E_c * de_E_c)
        l_power = np.power(1 - E_c, (1 - bf_E_c) * de_1_E_c)
        
        if True in np.isnan(l_power) or True in np.isnan(E_c_h):
            l_power = np.nan_to_num(l_power, copy=False, nan=0.0)
        E_c_l = 1 - l_power
        
        if True in np.isnan(E_c_h) or True in np.isnan(E_c_l):
            E_c_h = np.nan_to_num(E_c_h, copy=False, nan=0.0)
            E_c_l = np.nan_to_num(E_c_l, copy=False, nan=0.0)
        
        # T_c = np.where(E_c <= bf_E_c,
        #                E_c_l,
        #                E_c_h)
        
        T_c = np.where(E_c != 0,
                       np.where(E_c <= bf_E_c,
                                E_c_h,
                                E_c),
                       
                       np.where(E_c != 1,
                                np.where(E_c > bf_E_c,
                                         E_c_l,
                                         E_c),
                                E_c))
        
        if T_c.min() < 0.0:
            T_c = T_c + np.abs(T_c.min()) # Remove negative value
        
        # cv2.imwrite("T_c.hdr", T_c)
        # cv2.imwrite("T_c.jpg", T_c * 255.0)
        # cv2.imwrite("ori.jpg", image * 255.0)

        return T_c
        
    @staticmethod
    def tonemap_operator(hdr_image, mu=1000):
        tonemap_hdr = np.log(1 + mu * hdr_image) / np.log(1 + mu)
        return tonemap_hdr
    
    @staticmethod
    def inverse_tonemap_operator(tonemap_image, mu=1000):
        reverse_image = ((np.exp(tonemap_image) * (1 + mu)) - 1 ) / mu
        return reverse_image
    
    @staticmethod
    def lights_metering(hdr_image, metering_kernel_size = 51):
        # Metering kernel size should be odd number
        gaussion_filter = np.zeros((metering_kernel_size, metering_kernel_size))
        gaussion_filter[int(metering_kernel_size / 2), int(metering_kernel_size / 2)] = 1 # Center of kernel = 1
        gaussion_filter = cv2.GaussianBlur(gaussion_filter, (metering_kernel_size, metering_kernel_size), 0)
        
        Y_hdr = np.mean(hdr_image, axis=2) # Get the brightness of HDR image
        # Y_clip = np.where(Y_hdr > 1.0, 1.0, Y_hdr)
        
        for _ in range(2):
            Y_hdr = np.expand_dims(Y_hdr, axis=0)
            gaussion_filter = np.expand_dims(gaussion_filter, axis=0)
        
        patch_brightness = torch.nn.functional.conv2d(torch.from_numpy(Y_hdr).to(torch.float32),
                                                      torch.from_numpy(gaussion_filter).to(torch.float32),
                                                      stride=metering_kernel_size,)
        
        patch_brightness = patch_brightness.numpy() # Transform to numpy array
        patch_brightness = np.squeeze(patch_brightness) # Remove useless dim
        mean_brightness = np.mean(patch_brightness) # Get mean value of patch brightness
        
        correction_ev = np.log2(0.5 / mean_brightness) # Get correction value
        correction_hdr = hdr_image * pow(2, correction_ev) # Apply correction value
        return correction_hdr

    @staticmethod
    def float32_to_uint32(image):
        max_uint32 = np.iinfo(np.uint32).max
        image_uint32 = (image * max_uint32).astype(np.uint32)
        return image_uint32

    @staticmethod
    def histogram_equalization_float(tone_hdr, tone_hdr_clip, bit_depth=16):
        # image range [0, 1]
        # max_val = np.power(2, bit_depth)
        max_val = 1024
        q_k = max_val
        q_o = 5
        
        tone_hdr_ycbcr = cv2.cvtColor(tone_hdr, cv2.COLOR_BGR2YCrCb)
        tone_hdr_clip_ycbcr = cv2.cvtColor(tone_hdr_clip, cv2.COLOR_BGR2YCrCb)
        
        tone_hdr_ycbcr = tone_hdr_ycbcr * (q_k - 1)
        tone_hdr_clip_ycbcr = tone_hdr_clip_ycbcr * (q_k - 1)
        
        
        if bit_depth == 16:
            tone_hdr_ycbcr = tone_hdr_ycbcr.astype(np.uint16)
            tone_hdr_clip_ycbcr = tone_hdr_clip_ycbcr.astype(np.uint16)
            
        elif bit_depth == 8:
            tone_hdr_ycbcr = tone_hdr_ycbcr.astype(np.uint8)
            tone_hdr_clip_ycbcr = tone_hdr_clip_ycbcr.astype(np.uint8)    

        else:
            tone_hdr_clip_ycbcr = tone_hdr_clip_ycbcr.astype(np.uint32)
            tone_hdr_clip_ycbcr = tone_hdr_clip_ycbcr.astype(np.uint32)
        
        hist, bins = np.histogram(tone_hdr_ycbcr[:, :, 0].ravel(), bins=max_val, range=[0, max_val])
        cdf = np.cumsum(hist)

        # Get the equalization image
        
        he_curve = ((q_k - q_o) / (tone_hdr_ycbcr.shape[0] * tone_hdr_ycbcr.shape[1]) * cdf + q_o) / (q_k) # Using GT's histogram
        
        equalization_image = dataset.apply_crf(tone_hdr, he_curve)
        equalization_image_clip = dataset.apply_crf(tone_hdr_clip, he_curve)
        
        return equalization_image, equalization_image_clip, he_curve
        
    @staticmethod
    def find_curve(ori_img, apply_img, num_points=1024):
        # ori_img_gray = ori_img.mean(axis=2, keepdims=True)
        # apply_img_gray = apply_img.mean(axis=2, keepdims=True)
        
        # original_flat = ori_img_gray.flatten()
        # apply_flat = apply_img_gray.flatten()

        original_flat = ori_img.flatten()
        apply_flat = apply_img.flatten()
        
        # Sort the original image bu index
        # sort_indices = np.argsort(original_flat)
        # original_flat = original_flat[sort_indices]
        # apply_flat = apply_flat[sort_indices]
        
        # Remove the same value
        unique_original_values, inverse_idx = np.unique(original_flat, return_inverse=True)
        average_transformed_values = np.bincount(inverse_idx, weights=apply_flat) / np.bincount(inverse_idx)

        unique_original_values = np.append(unique_original_values, [0.0, 1.0])
        average_transformed_values = np.append(average_transformed_values, [0.0, average_transformed_values.max()])
        
        # Interpolate the curve
        evenly_spaced_original = np.linspace(0, 1, num_points)
        evenly_spaced_transformed = interp1d(unique_original_values, average_transformed_values)(evenly_spaced_original)
        
        return evenly_spaced_transformed 
    
    # ----- New -----
    @staticmethod
    def normalize_hdr_image(image):
        """Normalize an HDR image to [0, 1] range."""
        return image / image.max()
    
    @staticmethod
    def simple_tone_mapping(image, gamma=2.2):
        """Apply simple gamma correction for tone mapping."""
        return np.power(image, 1/gamma)
    # ----- New -----
    
def eval_image(images, log = None, max_value = None, file_name = None,mu = 10): # input shape (b, 3, H, W)
    if images.shape[1] == 1: # Gray image
        images = torch.cat((images, images, images), dim=1) # Gray to color domai
        
    if(log == True):
        images = torch.log(torch.add(mu * images, 1)) / torch.log(torch.tensor(1 + mu))
        
    if max_value != None:
        for _ in range(3):
            max_value = torch.unsqueeze(max_value, dim=1)
        images = images / max_value
    
    # After Processing
    if images.shape[0] < 4: # Make sure images have specific shape
        images = torch.cat((images, torch.zeros(tuple([4] + list(images.shape[1:]))).to("cuda")))
    
    images = torch.split(images, 1, dim=0) # Get batch first batch image (1, 3, H, W)
        
    concat_image = torch.cat((torch.cat((images[0], images[1]), dim=2),
                                torch.cat((images[2], images[3]), dim=2)), dim=3)

    concat_image = torch.squeeze(concat_image) # Remove first dim -> (3, H, W)
    if file_name != None:
        save_hdr_image(concat_image, file_name)
    concat_image = torch.split(concat_image, 1, dim=0) # BGR -> RGB
    concat_image = torch.cat((concat_image[2], concat_image[1], concat_image[0]), dim=0)

    return concat_image # For tensorboard add_image

def save_hdr_image(hdr_image, file_name): # Input_shape: (1, 3, H, W), BGR
    hdr_image = torch.squeeze(hdr_image) # Remove first dim -> (3, H, W)
    hdr_image = torch.permute(hdr_image, (1, 2, 0)) # (3, H, W) -> (H, W, 3)
    hdr_image = hdr_image.detach().cpu().numpy()
    
    if ".jpg" in file_name: # For jpg format
        cv2.imwrite(file_name, hdr_image * 255.0)
        return
    cv2.imwrite(file_name, hdr_image)
    return

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
def plot_curve(pd_curve, gt_curve):
    image_list = []
    pd_curve = torch.split(pd_curve.cpu(), 1, dim=0)
    gt_curve = torch.split(gt_curve.cpu(), 1, dim=0)
    for idx in range(len(pd_curve)):
        plt.figure()
        
        x = np.linspace(0, 1, 1024)
        
        plt.plot(x, torch.squeeze(pd_curve[idx]).detach().numpy(), label="PD")
        plt.plot(x, torch.squeeze(gt_curve[idx]).detach().numpy(), label="GT")
        plt.legend()

        # Convert the plot to a numpy array
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))

        # Convert the numpy array to a PyTorch tensor
        image = torch.from_numpy(image)
        
        image_list.append(image)
        plt.close()
        
        if idx >= 4:
            break
    
    concat_image = torch.cat((torch.cat((image_list[0], image_list[1]), dim=0),
                              torch.cat((image_list[2], image_list[3]), dim=0)), dim=1)
    concat_image = torch.squeeze(concat_image) # Remove first dim -> (H, W, 3)
    concat_image = torch.permute(concat_image, (2, 0, 1)) # (H, W, 3) -> (3, H, W)
    return concat_image

if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    aug = False
    batch_size = 16
    val_path = []
    val_root = "/work/u8083200/Thesis/datasets/SingleHDR_training_data/HDR-Real"
    for root, dirs, files in os.walk(val_root):
        files.sort()
        for file in files:
            if(".hdr" in file):
                val_path.append(os.path.join(root, file))
                continue
    
    # val_path = ["/work/u8083200/Thesis/datasets/SingleHDR_training_data/HDR-Real/HDR_gt/00008.hdr"]
    val_path = ["/work/u8083200/Thesis/datasets/SingleHDR_training_data/HDR-Real/HDR_gt/07460.hdr"]
    val_dataloader = DataLoader(dataset(val_path, stage=4, image_size=512, aug=aug, sigma=2.0), shuffle=True, batch_size=batch_size)
    
    for idx, data in enumerate(val_dataloader):
        x = x + 1
        pass