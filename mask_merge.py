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

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(img_hsv, np.array([0, 50, 40]), np.array([9, 220, 255]))
    mask2 = cv2.inRange(img_hsv, np.array([10, 130, 90]), np.array([14, 210, 150]))

    mask = cv2.bitwise_or(mask1, mask2)

    cv2.imshow("mask1", mask1)
    cv2.imshow("mask2", mask2)
    cv2.imshow("mask+", mask)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    apple = 1
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

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
