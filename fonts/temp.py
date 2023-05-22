from s3_helper import S3Utils
import requests
import json
from PIL import Image


url = "http://127.0.0.1:8003/lama"


s3_obj = S3Utils()

def run(input_image, text):
    input_image.save("/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/temp/1.png")
    # mask, polygon = main("/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/temp/1.png")

    upload_img_path = "text_removal/127_532/img/1.png"
    upload_mask_path = "text_removal/127_532/mask/1_mask001.png"

    s3_obj.write_image_to_s3(input_image, upload_img_path)
    s3_obj.write_image_to_s3(input_image, upload_mask_path)

    payload = json.dumps({
        "updated_masks_path": "text_removal/127_532/mask/",
        "image_generator_result_path": "text_removal/127_532/img/"
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    text_removed_path = json.loads(response.text)["inpainted_image_path"]
    text_removed_img = s3_obj.read_image_from_s3(text_removed_path + "/1.png")

run(Image.open("/home/ubuntu/tausif_workspace/text_mask/DPText-DETR/1.jpg"), "text")
        