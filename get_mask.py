import cv2
import numpy as np


class Image_mask:
    def __init__(self, image, in_image, coords):
        try:
            # Load the input image
            print(f'Loading the image {image}')
            self.image = cv2.imread(image[0])
            self.in_image = cv2.imread(in_image)
            print(type(self.image))
        except Exception as e:
            print(e)
            self.image = image
        self.vertices = self.convert2poly(coords)
        

    def convert2poly(self, coords):
        n_vertices = []
        print(len(coords))
        for coord in coords:
            print(len(coord))
            coord = list(coord)
            # Convert the flat list of coordinates to a list of tuples of (x,y) coordinate pairs
            vertices = [(np.round(coord[i].cpu().numpy()), np.round(coord[i+1].cpu().numpy())) for i in range(0, len(coord), 2)]
            # Append the first vertex to the end to close the polygon
            n_vertices.append(vertices)
        return n_vertices

    
    def get_mask(self, out_name, train=False):
        polygon_vertices = [np.array(ver, dtype=np.int32) for ver in self.vertices]
        # print(polygon_vertices[1])
        # Create a mask with the same size as the image, and fill it with zeros
        # print(type(self.image))
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # Draw the polygons on the mask with white color (255)
        cv2.fillPoly(mask, polygon_vertices, 255)
        # Apply the mask to the input image using bitwise AND, if provide
        if train:
            out_train = 'gt_masks/'+out_name.split('/')[-1]
            cropped_image_train = cv2.bitwise_and(self.in_image, self.in_image, mask=mask)
            # cv2.imwrite('res_trn.png', result_train)
            cv2.imwrite(out_train, cropped_image_train)
        # Apply the mask to the image using bitwise AND
        cropped_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        cv2.imwrite('real_data/'+out_name.split('/')[-1], cropped_image)
        # Create a black background image with the same size as the input image
        background = np.zeros_like(self.image)
        # Draw the polygons on the background with white color (255)
        cv2.fillPoly(background, polygon_vertices, 255)
        # Apply the mask to the background using bitwise AND
        background = cv2.bitwise_and(background, background, mask=mask)
        # Combine the cropped image and the background image using bitwise OR
        result = cv2.bitwise_or(cropped_image, background)
        # Save the result image
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_name, result)
        return result