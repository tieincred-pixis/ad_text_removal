import gradio as gr
from inference import main
from s3_helper import S3Utils
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm


url = "https://5ce4-119-82-102-194.in.ngrok.io/lama"


s3_obj = S3Utils()


def adjust_text_size(text, font_path, box_width, box_height):
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    # while font.getsize(text)[0] > box_width or font.getsize(text)[1] > box_height:
    #     font_size -= 1
    # font = ImageFont.truetype(font_path, font_size)
    return font

def convert2poly(coords):
    n_vertices = []
    for coord in coords:
        print(len(coord))
        coord = list(coord)
        # Convert the flat list of coordinates to a list of tuples of (x,y) coordinate pairs
        vertices = [(np.round(coord[i].cpu().numpy()), np.round(coord[i+1].cpu().numpy())) for i in range(0, len(coord), 2)]
        # Append the first vertex to the end to close the polygon
        n_vertices.append(vertices)
    return n_vertices

def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we traverse the collection of points only once, 
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [bot_left_x, top_right_y, top_right_x-bot_left_x, top_right_y-bot_left_y]
# def bounding_box(points):
#     x_coordinates, y_coordinates = zip(*points)
#     return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def write_text_in_boxes(bounding_boxes, img, font_path, text, color=(255,0,0)):
    with img as img:
        draw = ImageDraw.Draw(img)
        c=0
        print
        for box in tqdm(bounding_boxes):
            c+=1
            font = adjust_text_size(text, font_path, box[2]-box[0], box[3]-box[1])

            text_x = (box[0] + box[2]) // 2
            text_y = (box[1] + box[3]) // 2
            if c > 2:
                break
            draw.text((text_x, text_y), text, fill=color, font=font, anchor='mm')
        img.save('out_written.png')
        return img
        

def run(input_image, text):
    input_image.save("/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/temp/1.png")
    mask, polygon = main("/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/temp/1.png")
    print(polygon)

    upload_img_path = "text_removal/127_531/img/1.png"
    upload_mask_path = "text_removal/127_531/mask/1_mask001.png"
    # input_image = input_image.resize((700,700))
    mask = mask.resize(input_image.size)
    mask.save("temp_mask.png")

    s3_obj.write_image_to_s3(input_image, upload_img_path)
    s3_obj.write_image_to_s3(mask, upload_mask_path)

    payload = json.dumps({
        "updated_masks_path": "text_removal/127_531/mask/",
        "image_generator_result_path": "text_removal/127_531/img/"
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    text_removed_path = json.loads(response.text)["inpainted_image_path"]
    print(text_removed_path)
    text_removed_img = s3_obj.read_image_from_s3(text_removed_path + "/1_mask001.png")

    polygon_ = convert2poly(polygon)
    print(polygon)
    polygon = []
    for poli in polygon_:
        polygon.append(bounding_box(poli))
    print(polygon)
    print(text)
    text_removed_img = write_text_in_boxes(polygon, text_removed_img, "/home/ubuntu/tausif_workspace/text_mask/fonts/Black Light.ttf", text)
    print(polygon)
    text_removed_img.save('final_out.png')
    return [text_removed_img]

block = gr.Blocks(css="footer {visibility: hidden", theme='freddyaboulton/test-blue').queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Text Removal")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="pil", label="Input Image")
            Text = gr.Textbox(label="Text")
            run_button = gr.Button(label="Submit")
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")
        run_button.click(fn=run, inputs=[input_image, Text,], outputs=[gallery])



block.launch(server_name='0.0.0.0',server_port=8004, show_api=False, share = False)