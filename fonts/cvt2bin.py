from PIL import Image
import os
import subprocess
from tqdm import tqdm

def run_dptext_detr(config_file, input_image, output_image, model_weights, maskname):
    script_path = "demo/demo.py"
    command = [
        "python", script_path,
        "--config-file", config_file,
        "--input", input_image,
        "--output", output_image,
        "--maskname", maskname,
        "--opts", "MODEL.WEIGHTS", model_weights
    ]
    
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def convert_to_binary_image(input_image_path, output_image_path, threshold=128):
    # Open the input image
    image = Image.open(input_image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Apply the threshold and convert to binary image
    binary_image = grayscale_image.point(lambda x: 0 if x < threshold else 255, mode='1')

    # Save the binary image
    binary_image.save(output_image_path)

if __name__ == '__main__':
    config_file_path = "configs/DPText_DETR/TotalText/R_50_poly.yaml"
    model_weights_path = "weights/pretrain.pth"
    images = os.listdir('dataset_new')[:5]
    for image in tqdm(images):
        input_image_path = os.path.join('dataset_new', image)
        output_image_path = input_image_path
        threshold_value = 128
        # convert_to_binary_image(input_image_path, output_image_path, threshold_value)
        out_name = image.split('.')[0]
        output_image_path = f'output/{out_name}'+'.png'
        maskout = f'masks/{out_name}'+'.png'
        try:
            run_dptext_detr(config_file_path, input_image_path, output_image_path, model_weights_path, maskout)
        except Exception as e:
            print(e)
            pass

# python demo/demo.py --config-file configs/DPText_DETR/TotalText/R_50_poly.yaml --input 1.jpg --output output/1.png --maskname masks/1.png --opts MODEL.WEIGHTS pretrain.pth
