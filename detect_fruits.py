import cv2
import json
import click
import numpy as np

from glob import glob
from tqdm import tqdm

from typing import Dict


def color_profiles(n):
    if n == 0:
        name = "apple"
        hsv_lower = np.array([0, 50, 50])
        hsv_upper = np.array([9, 255, 255])
        return name, hsv_lower, hsv_upper
    if n == 1:
        name = "banana"
        hsv_lower = np.array([24, 75, 75])
        hsv_upper = np.array([35, 255, 255])
        return name, hsv_lower, hsv_upper
    if n == 2:
        name = "orange"
        hsv_lower = np.array([12, 50, 200])
        hsv_upper = np.array([20, 255, 255])
        return name, hsv_lower, hsv_upper


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
    img = cv2.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    apple = 0
    banana = 0
    orange = 0

    for i in range(3):
        name, hsv_lower, hsv_upper = color_profiles(i)
        mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
        cv2.imshow(f"Mask {i}", mask)
        conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = sorted(conts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.boundingRect(biggest)
        x, y, w, h = rect

        if w < 100 or h < 100:
            continue

        # if i == 0:
        #     len(biggest) = apple
        # elif i == 1:
        #     len(biggest) = banana
        # elif i == 2:
        #     len(biggest) = orange

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)

    cv2.imshow("Post", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {'apple': apple, 'banana': banana, 'orange': orange}


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
