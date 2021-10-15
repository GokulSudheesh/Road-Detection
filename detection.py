import cv2
import argparse
from Lane import Lane


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='Path to input video', required=True)
    parser.add_argument('--dim', type=int, nargs="+", help='Width and height of the output frame')

    return parser.parse_args()


def get_predictions(width=640, height=480, *, input_dir):
    cap = cv2.VideoCapture(input_dir)
    lane = Lane(width, height)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        new_img = lane.road_lanes(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), image_layer=image)
        cv2.imshow("Test", new_img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = init_args()

    if args.dim:
        get_predictions(*args.dim, input_dir=args.video_dir)
    else:
        get_predictions(input_dir=args.video_dir)
