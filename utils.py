import torch
import cv2
import numpy as np

def transfer_to_cuda_and_float32(tensor_dict):
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].to(torch.device('cuda')).float()
    return tensor_dict

def merge_hdr(prediction: dict):
    # Convert all images in the dictionary
    opencv_images = [convert_to_opencv_format(img) for img in prediction["images"]]
    times = np.array(prediction['times'])
    # Update the dictionary with converted images
    merge_Robertson = cv2.createMergeRobertson()

    hdr_image = merge_Robertson.process(opencv_images, times)
    return torch.tensor(np.transpose(hdr_image, (2, 0, 1)))

def convert_to_opencv_format(image):
    image = image.squeeze(0)  # Remove the batch dimension
    image = np.transpose(image.cpu().detach().numpy(), (1, 2, 0))  # Transpose to (H, W, 3)
    return np.uint8(image * 255)  # Convert to uint8
