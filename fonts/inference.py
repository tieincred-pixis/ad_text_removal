import os
import pdb
import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from cvt2bin import run_dptext_detr
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

def enhance(image):
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced_image = contrast_enhancer.enhance(4.0)
    return enhanced_image

def erode(cycles, image, pixel):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(pixel))
    return image


def dilate(cycles, image, pixel):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(pixel))
    return image

def make_inference(modelT, modelM,config_file_path,input_image, enhance_contrast=False):
    out_name = input_image.split('/')[-1].split('.')[0]
    output_image_path = f'output/{out_name}'+'.png'
    maskout = f'masks/{out_name}'+'.png'
    # pdb.set_trace()
    try:
        run_dptext_detr(config_file_path, input_image, output_image_path, modelT, maskout)
    except Exception as e:
        print(e)
        pass
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = smp.Unet('resnet50', classes=1, encoder_weights='imagenet').to(device)
    model.load_state_dict(torch.load(modelM, map_location=device))
    model.eval()

    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_folder = maskout
    image = image_folder.split('/')[-1]
    img = Image.open(image_folder).convert('RGB')
    if enhance_contrast:
        img = enhance(img)
    img = transform(img).unsqueeze(0)

    # Move the image to the device
    img = img.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(img)
        # Apply sigmoid function to normalize the output to [0, 1]
        output = torch.sigmoid(output)

    # Convert the output to a PIL image and save it
    output = output.cpu().squeeze().numpy()
    output = (output > 0.1).astype('uint8') * 255
    # output = (output > 0.01).astype('uint8') * 255
    output = Image.fromarray(output, mode='L')
    # eroded_image = erode(1,output,9)
    dilated_image = dilate(2,output,5)
    # eroded_image = erode(1,dilated_image,15)
    
    # dilated_image.save('dilated_out.png')
    # eroded_image.save('eroded_out.png')
    output.save('out.png')
    return dilated_image, torch.load('polygons.pt')

def main(input_image):
    config_file_path = "configs/DPText_DETR/TotalText/R_50_poly.yaml"
    modelT = "weights/pretrain.pth"
    modelM = "/home/ubuntu/tausif_workspace/text_mask/model_weights_50.pth"
    mask, polygon = make_inference(modelT, modelM, config_file_path, input_image)
    return mask, polygon


# main('1.jpg')