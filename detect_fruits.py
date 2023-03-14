import cv2
import json
import click
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict


class Fruit:

    def __init__(self, img, img_hsv, name):
        self.img = img
        self.img_hsv = img_hsv
        self.name = name
        self.mask = None
        self.found_objects = 0

    # def img_conv2hsv(self):
    #     """Image resize and conv to hsv"""
    #
    #     img_res = cv2.resize(self.img_hsv, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    #     img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
    #
    #     return img_hsv

    def create_mask(self, hsv_lower, hsv_upper):
        """Create a mask with given hsv threshold values"""

        for lower, upper in zip(hsv_lower, hsv_upper):
            if self.mask is not None:
                self.mask = cv2.bitwise_or(self.mask, cv2.inRange(self.img_hsv, lower, upper))
            else:
                self.mask = cv2.inRange(self.img_hsv, hsv_lower[0], hsv_upper[0])

        return self.mask

        # if self.name == "apple":
        #     mask1 = cv2.inRange(self.img_hsv, hsv_lower[0], hsv_upper[0])
        #     mask2 = cv2.inRange(self.img_hsv, hsv_lower[1], hsv_upper[1])
        #     mask3 = cv2.inRange(self.img_hsv, hsv_lower[2], hsv_upper[2])
        #
        #     mask = cv2.bitwise_or(mask1, mask2)
        #     self.mask = cv2.bitwise_or(mask, mask3)
        #     return self.mask
        #     # self.mask = mask1 + mask2 + mask3
        # else:
        #     self.mask = cv2.inRange(self.img_hsv, self.hsv_lower, self.hsv_upper)
        #     return self.mask

    # def find_conts(self):
    #     """Finds contours on a given mask and returns the biggest one"""
    #
    #     mask = self.mask
    #     conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # biggest = sorted(conts, key=cv2.contourArea, reverse=True)[0]
    #
    #     return conts

    def get_rect(self, mask):
        """
        Finds contours on a given mask, finds rectangles around these contours
        and returns dimensions of found_objects rectangles
        """

        conts, _hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for cont in conts:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect

            if w < 70 or h < 70:
                continue

            rects.append(rect)
            self.found_objects += 1

        return rects

    def draw_rect(self, rects):
        """

        Draws a rectangle around found objects on original image

        Parameters
        ----------
        opencv object containing rectangles from function get_rect

        Returns
        -------
        None

        """

        if rects:
            for rect in rects:
                x, y, w, h = rect
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                cv2.putText(self.img, self.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
                            thickness=2)

        # mask_res = cv2.resize(self.mask, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(f"Mask of {self.name.upper()}", self.mask)


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

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_res = cv2.resize(img, dsize=None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)

    apple = Fruit(img=img_res, img_hsv=img_hsv, name="apple")
    banana = Fruit(img=img_res, img_hsv=img_hsv, name="banana")
    orange = Fruit(img=img_res, img_hsv=img_hsv, name="orange")

    apple_mask = apple.create_mask(hsv_lower=np.array([[0,  50,  40],  [0,  75,  75], [130,  25,  40]]),
                                   hsv_upper=np.array([[9, 220, 255], [18, 215, 150], [180, 200, 230]]))

    banana_mask = banana.create_mask(hsv_lower=np.array([[20,  90, 120]]),
                                     hsv_upper=np.array([[30, 255, 250]]))

    orange_mask = orange.create_mask(hsv_lower=np.array([[10, 200, 130]]),
                                     hsv_upper=np.array([[18, 255, 255]]))

    apple_rects  = apple.get_rect(apple_mask)
    banana_rects = banana.get_rect(banana_mask)
    orange_rects = orange.get_rect(orange_mask)

    # apple.draw_rect(apple_rects)
    # banana.draw_rect(banana_rects)
    # orange.draw_rect(orange_rects)
    #
    # cv2.imshow("Post-mask", img_res)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return {'apple': apple.found_objects, 'banana': banana.found_objects, 'orange': orange.found_objects}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
