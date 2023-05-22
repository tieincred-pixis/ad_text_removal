import requests
import time
import os
import json
import io
from io import BytesIO
import base64
import numpy as np
import cv2
import ast
from inference import main
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, request, Response, make_response
from flask_cors import CORS
from s3_helper import S3Utils
import warnings
from flask import Flask


url = "http://f7ee-106-51-83-192.ngrok-free.app/lama"
url_font = 'http://192.168.80.159:5000/detect_font'


s3_obj = S3Utils()
app = Flask(__name__)
CORS(app)

heights = []
fonts = []
def convert_coordinates(x, y, width, height):
    half_width = width / 2
    half_height = height / 2

    new_x = x + half_width
    new_y = y + half_height

    return new_x, new_y

def hex_to_rgb(hexcode):
    # remove the hash symbol if present
    hexcode = hexcode.lstrip('#')

    # convert hexcode to RGB values
    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)

    # return as a tuple
    return (r, g, b)

def scale_bbox(bbox, input_image, dim):
    # Get the dimensions of the input image
    input_width, input_height = input_image.size
    dim = [float(i) for i in dim.strip('[]').split(',')]
    # Calculate the scaling factors for width and height
    width_scale = input_width / dim[0]
    height_scale = input_height / dim[1]

    # Scale the bounding box coordinates
    x1 = int(bbox[0] * width_scale)
    y1 = int(bbox[1] * height_scale)
    x2 = int(bbox[2] * width_scale)
    y2 = int(bbox[3] * height_scale)

    # Return the scaled bounding box
    return [x1, y1, x2, y2]

def update_image(og_image, image, bboxes, dim):
    print(bboxes)
    # Convert image to numpy array
    bboxes = ast.literal_eval(bboxes)
    img = np.array(image)

    # Create a black image of the same shape
    black_img = np.zeros_like(img)
    
    # Copy the bounding box area from original image to black image
    for bbox in bboxes:
        bbox = scale_bbox(bbox, image, dim)
        x1, y1, x2, y2 = bbox
        print("**************")
        print(bbox)
        black_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        fonts.append(get_font_from_bbox(og_image, bbox))
        heights.append(abs(y2 - y1 - 2))
    

    # Convert back to PIL Image and return
    return Image.fromarray(black_img)


# def get_font_from_bbox(image, bbox):
#     # Convert image to numpy array
#     img = np.array(image)
#     # Extract bounding box coordinates
#     x1, y1, x2, y2 = bbox
#     # Crop image using bounding box
#     cropped_img = img[y1:y2, x1:x2]
#     # Convert back to PIL Image
#     cropped_img_pil = Image.fromarray(cropped_img)
#     # Use detect_font function to get font
#     # Define the files parameter for the POST request
#     files = {'file': file}
#     # Send the POST request
#     response = requests.post(url_font, files=files)
#     print(response)
#     return response['font']
    # return 'arial'
def get_font_from_bbox(image, bbox, url_font=url_font):
    # Convert image to numpy array
    img = np.array(image)

    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox
    print(bbox)
    # Crop image using bounding box
    cropped_img = img[y1:y2, x1:x2]

    # Convert back to PIL Image
    cropped_img_pil = Image.fromarray(cropped_img)
    cropped_img_pil.save('cropped.jpg')
    # payload = {}
    # files=[
    # ('file',('cropped.jpg',open('/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/cropped.jpg','rb'),'image/jpeg'))
    # ]
    # headers = {}
    # print('calling API')
    # response = requests.request("POST", url_font, headers=headers, data=payload, files=files)
    # print('Done')
    # print(response.text)
    return 'arial'

@app.route('/process-image', methods=['POST'])
def process_image():
    file1 = request.files['input_image']
    img = Image.open(file1)
    bboxes = request.form['bboxes']
    print('print Image Dimension')
    img_dim = request.form['dimension']
    print(img_dim)
    print(bboxes)
    img.save("temp/1.png") 
    mask, polygon = main("temp/1.png")
    # Process image
    folder_name = str(time.time()).replace('.','')
    upload_img_path = f"text_removal/{folder_name}/img/1.png"
    upload_mask_path = f"text_removal/{folder_name}/mask/1_mask001.png"
    mask = mask.resize(img.size)
    print('going inside mask')
    mask = update_image(img, mask, bboxes, img_dim)
    mask.save("temp_mask.png")
    print('going outside mask')
    s3_obj.write_image_to_s3(img, upload_img_path)
    s3_obj.write_image_to_s3(mask, upload_mask_path)

    payload = json.dumps({
        "updated_masks_path": os.path.dirname(upload_mask_path)+'/',
        "image_generator_result_path": os.path.dirname(upload_img_path)+'/'
    })
    print(payload)
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    text_removed_path = json.loads(response.text)["inpainted_image_path"]
    print(text_removed_path)
    text_removed_img = s3_obj.read_image_from_s3(text_removed_path + "/1_mask001.png")
    
    # Encode image to JPEG format
    # buffer = BytesIO()
    # text_removed_img.save(buffer, format='JPEG')
    # # img.save(buffer, format='JPEG')
    # buffer.seek(0)
    resp = make_response()
    buffer = BytesIO()
    text_removed_img.save(buffer, format='JPEG')
    buffer.seek(0)

    # Encode image data to base64
    # image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # # Return image and folder name as a JSON object
    # return jsonify({
    #     'image': image_base64,
    #     'folder_name': folder_name,
    # })
    # Set the content type and encoding for the image
    resp.headers.set('Content-Type', 'image/jpeg')
    resp.headers.set('Content-Disposition', 'attachment', filename='image.jpg')
    
    # Set the content type and encoding for the text
    resp.headers.set('Content-Type', 'text/plain')
    
    # Combine the image and text data into the response
    resp.set_data(buffer.read() + b'\n\n' + folder_name.encode())
    # Return image as a response
    return resp


@app.route('/fix-images', methods=['POST'])
def fix_images():
    print('Hitting fix image')
    image = request.files['input_image']
    mask = request.files['mask']
    folder_name = "request.form['folder_name']"
    img = Image.open(image)
    mask = Image.open(mask)

    mask = mask.resize(img.size)

    # Process image
    upload_img_path = f"text_removal/{folder_name}/img_up/1.png"
    upload_mask_path = f"text_removal/{folder_name}/mask_up/1_mask001.png"

    s3_obj.write_image_to_s3(img, upload_img_path)
    s3_obj.write_image_to_s3(mask, upload_mask_path)

    # Process images
    payload = json.dumps({
        "updated_masks_path": os.path.dirname(upload_mask_path)+'/',
        "image_generator_result_path": os.path.dirname(upload_img_path)+'/'
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    text_removed_path = json.loads(response.text)["inpainted_image_path"]
    print(text_removed_path)
    text_removed_img = s3_obj.read_image_from_s3(text_removed_path + "/1_mask001.png")
    
    # Encode image to JPEG format
    # Encode image to JPEG format
    buffer = BytesIO()
    text_removed_img.save(buffer, format='JPEG')
    buffer.seek(0)

    return Response(buffer, mimetype='image/jpeg')
    image_encoded = base64.b64encode(buffer.getvalue()).decode()
    # Return image, fonts and font_sizes as a response
    return jsonify({
        'image': image_encoded
    })


@app.route('/on_next', methods=['GET'])
def get_data():
    return jsonify({'fonts': fonts, 'font_sizes': font_sizes})

@app.route('/upload-results', methods=['POST'])
def upload_results():
    if 'result_image' not in request.files:
        return {'error': 'No file found in the request'}, 400
    image = request.files['result_image']
    if image.filename == '':
        return {'error': 'No file selected'}, 400
    img = Image.open(image)

    # Process image
    image_name = str(time.time()).replace('.','')
    upload_img_path = f"text_removal/results_folder/{image_name}.png"
    try:
        s3_obj.write_image_to_s3(img, upload_img_path)
    except:
        return {'error': 'S3 writer failed'}, 400

    return {'message': 'File uploaded successfully'}, 200

@app.route('/add-text-to-image', methods=['POST'])
def add_text_to_image():
    print('HITTING ADDING IMAGE')
    # Load image from file
    image_file = request.files['image']
    image = Image.open(image_file)
    dim = request.form['dimension']
    # Get the original image size
    orig_width, orig_height = image.size
    
    # Load JSON data from request payload
    loaders = ['bboxes', 'texts', 'fonts', 'font_sizes', 'color']
    data = {}
    for loader in loaders:
        print(request.form[loader])
        data[loader] = ast.literal_eval(request.form[loader])

    print(len(data['bboxes']))
    dim = [float(i) for i in dim.strip('[]').split(',')]
    # Calculate the scaling factor
    scale_width =  orig_width / dim[0]
    scale_height = orig_height / dim[1]
    
    # Iterate through bounding boxes and add text to the image
    draw = ImageDraw.Draw(image)
    for i in range(len(data['bboxes'])):
        bbox_x, bbox_y = data['bboxes'][i]
        # bbox_x, bbox_y = update_coordinate((bbox_x, bbox_y),dim)
        print('*************')
        print(bbox_x, bbox_y)
        print(dim)
        print('*************')
        bbox_x = int(bbox_x * scale_width)
        bbox_y = int(bbox_y * scale_height)
        text = data['texts'][i]
        # font = 'fonts/'+data['fonts'][i]
        # just to test
        font = 'fonts/times new roman.ttf'
        font_size = int(data['font_sizes'][i] * scale_height)
        color = hex_to_rgb(data['color'][i].replace('(','').replace(')',''))
        print(font)
        font = ImageFont.truetype(font, font_size)
        # bbox_x, bbox_y = convert_coordinates(bbox_x, bbox_y, dim[0], dim[1])
        print('+++++++++++++++')
        print(bbox_x, bbox_y)
        draw.text((bbox_x, bbox_y), text, font=font, fill=color)
    
    # Encode image to JPEG format
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Return image as a response
    return Response(img_bytes, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8004)