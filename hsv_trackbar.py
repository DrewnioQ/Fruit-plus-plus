import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np


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

    def empty(i):
        pass

    cv2.namedWindow("image")

    def on_trackbar(val):
        hue_min = cv2.getTrackbarPos("Hue Min", "image")
        hue_max = cv2.getTrackbarPos("Hue Max", "image")
        sat_min = cv2.getTrackbarPos("Sat Min", "image")
        sat_max = cv2.getTrackbarPos("Sat Max", "image")
        val_min = cv2.getTrackbarPos("Val Min", "image")
        val_max = cv2.getTrackbarPos("Val Max", "image")

        lower = np.array([hue_min, sat_min, val_min])
        upper = np.array([hue_max, sat_max, val_max])

        mask = cv2.inRange(img_hsv, lower, upper)

        # cv2.imshow("Output1", img)
        # cv2.imshow("Output2", img_hsv)
        cv2.imshow("image", mask)
        cv2.imshow("original", img)

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.createTrackbar("Hue Min", "image", 0, 180, on_trackbar)
    cv2.createTrackbar("Hue Max", "image", 180, 180, on_trackbar)
    cv2.createTrackbar("Sat Min", "image", 0, 255, on_trackbar)
    cv2.createTrackbar("Sat Max", "image", 255, 255, on_trackbar)
    cv2.createTrackbar("Val Min", "image", 0, 255, on_trackbar)
    cv2.createTrackbar("Val Max", "image", 255, 255, on_trackbar)

    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv2.waitKey()
    cv2.destroyAllWindows()

    apple = 0
    banana = 0
    orange = 0

    return {'apple': apple, 'banana': banana, 'orange': orange}


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

    # with open(output_file_path, 'w') as ofp:
    #     json.dump(results, ofp)


if __name__ == '__main__':
    main()
