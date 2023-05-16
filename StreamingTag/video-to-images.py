import cv2
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-v", "--video_filename", required=True)
    parser.add_argument("-i", "--img_subpath", required=True)
    parser.add_argument("-l", "--limit", type=int, required=True)
    ARGS = parser.parse_args()
    video_filename = os.path.join(ARGS.path, ARGS.video_filename)
    img_path = os.path.join(ARGS.path, ARGS.img_subpath)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cap = cv2.VideoCapture(video_filename)
    for idx in range(ARGS.limit):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(img_path, f"{idx}.png"), frame)