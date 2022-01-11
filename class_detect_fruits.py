import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict


class Fruit:

    img = None

    def __init__(self, img_hsv, name, hsv_lower, hsv_upper):
        self.img_hsv = img_hsv
        self.name = name
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper

    def img_conv2hsv(self):
        """Image resize and conv to hsv"""

        img_res = cv2.resize(self.img_hsv, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)

        return img_hsv

    def create_mask(self):
        """Create a mask with given hsv threshold values"""

        img_hsv = self.img_conv2hsv()

        mask = cv2.inRange(img_hsv, self.hsv_lower, self.hsv_upper)

        return mask

    def find_conts(self):
        """Finds contours on a given mask"""

        mask = self.create_mask()

        conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = sorted(conts, key=cv2.contourArea, reverse=True)[0]

        return biggest

    def get_rect(self):
        biggest = self.find_conts()

        rect = cv2.boundingRect(biggest)
        x, y, w, h = rect

        if w < 100 or h < 100:
            return None

        return rect


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    # TODO: Implement detection method.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img_res = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    Fruit.img = img
    apple = Fruit(img_hsv=img_hsv,
                  name="apple",
                  hsv_lower=np.array([0, 50, 50]),
                  hsv_upper=np.array([9, 255, 255]))
    banana = Fruit(img_hsv=img_hsv,
                   name="banana",
                   hsv_lower=np.array([24, 75, 75]),
                   hsv_upper=np.array([35, 255, 255]))
    orange = Fruit(img_hsv=img_hsv,
                   name="orange",
                   hsv_lower=np.array([12, 50, 200]),
                   hsv_upper=np.array([20, 255, 255]))

    apple_rect = apple.get_rect()
    banana_rect = banana.get_rect()
    orange_rect = orange.get_rect()
    fruits_rects = [apple_rect, banana_rect, orange_rect]

    # for fruit_rect in fruits_rects:
    #     if fruit_rect:
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    #         cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
    #         mask_res = cv2.resize(mask, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    #         cv2.imshow(f"Mask {fruit_rect.create_mask()}", mask_res)
    img_res = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Post", img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {'apple': 0, 'banana': 0, 'orange': 0}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):
    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
