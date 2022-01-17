import cv2
import json
import click
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Dict


class Fruit:
    img = None

    def __init__(self, img_hsv, name):
        self.img_hsv = img_hsv
        self.name = name
        self.hsv_lower = np.array([])
        self.hsv_upper = np.array([])
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
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        # for lower, upper in zip(hsv_lower, hsv_upper):
        #     if mask is None:
        #         self.mask = cv2.inRange(self.img_hsv, hsv_lower[0], hsv_upper[0])
        #     else:
        #         self.mask = cv2.bitwise_or(self.mask, cv2.inRange())

        if self.name == "apple":
            mask1 = cv2.inRange(self.img_hsv, hsv_lower[0], hsv_upper[0])
            mask2 = cv2.inRange(self.img_hsv, hsv_lower[1], hsv_upper[1])
            mask3 = cv2.inRange(self.img_hsv, hsv_lower[2], hsv_upper[2])

            mask = cv2.bitwise_or(mask1, mask2)
            self.mask = cv2.bitwise_or(mask, mask3)
            # self.mask = mask1 + mask2 + mask3
        else:
            self.mask = cv2.inRange(self.img_hsv, self.hsv_lower, self.hsv_upper)

    # def find_conts(self):
    #     """Finds contours on a given mask and returns the biggest one"""
    #
    #     mask = self.mask
    #     conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # biggest = sorted(conts, key=cv2.contourArea, reverse=True)[0]
    #
    #     return conts

    def get_rect(self):
        """
        Finds contours on a given mask, finds rectangles around these contours
        and returns dimensions of found_objects rectangles
        """
        mask = self.mask
        conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []

        for cont in conts:
            rect = cv2.boundingRect(cont)
            x, y, w, h = rect

            if w < 70 or h < 70:
                continue

            rects.append(rect)
            self.found_objects += 1

        return rects

    def draw_rect(self):
        """Draws a rectangle around found objects on original image"""

        rects = self.get_rect()

        if rects:
            for rect in rects:
                x, y, w, h = rect
                cv2.rectangle(Fruit.img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                cv2.putText(Fruit.img, self.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255),
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
    # TODO: Implement detection method.
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_res = cv2.resize(img, dsize=None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)

    Fruit.img = img_res
    apple = Fruit(img_hsv=img_hsv, name="apple")
    banana = Fruit(img_hsv=img_hsv, name="banana")
    orange = Fruit(img_hsv=img_hsv, name="orange")

    apple.create_mask(hsv_lower=np.array([[0, 50, 40], [0, 75, 75], [130, 25, 40]]),
                      hsv_upper=np.array([[9, 220, 255], [18, 215, 150], [180, 200, 230]]))

    banana.create_mask(hsv_lower=np.array([20, 90, 120]),
                       hsv_upper=np.array([30, 255, 250]))

    orange.create_mask(hsv_lower=np.array([11, 200, 130]),
                       hsv_upper=np.array([18, 255, 255]))

    apple.draw_rect()
    banana.draw_rect()
    orange.draw_rect()

    cv2.imshow("Post-mask", img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
