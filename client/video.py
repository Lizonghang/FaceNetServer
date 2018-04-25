# coding=utf-8
import cv2
import numpy as np
import requests
import darknet
import StringIO
import argparse
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def crop_and_resize(im, bb):
    im = np.array(im)
    cropped = im[bb[1]:bb[3], bb[0]:bb[2], :]
    try:
        im = Image.fromarray(cropped, 'RGB')
    except ValueError:
        print bb
        print im.shape
        import sys
        sys.exit(0)
    im = im.resize((160, 160))
    return np.array(im)


def detect(net, meta, im):
    """im: opencv format image data"""
    im = darknet.array_to_image(im)
    darknet.rgbgr_image(im)
    return darknet.detect(net, meta, im)


def convert_size(fn_src, fn_dst, size=(640, 480)):

    cap = cv2.VideoCapture(fn_src)
    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(fn_dst, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        out.write(frame)
        if cv2.waitKey(1) == 27:
            break

    out.release()
    cap.release()


def capture(fn_dst):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow('player')

    ttfont = ImageFont.truetype('simsun.ttc', 20)

    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(fn_dst, fourcc, fps, cap.read()[1].shape[:2][::-1])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        im = Image.fromarray(frame)
        drawer = ImageDraw.Draw(im)

        drawer.text(
            (10, 10),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            fill=(0, 0, 255),
            font=ttfont
        )
        frame = np.array(im)

        cv2.imshow('player', frame)

        out.write(frame)

        if cv2.waitKey(1000 / fps) == 27:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main(fn_src, fn_dst):
    cap = cv2.VideoCapture(fn_src)
    ttfont = ImageFont.truetype('simsun.ttc', 20)

    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    out = cv2.VideoWriter(fn_dst, fourcc, fps, cap.read()[1].shape[:2][::-1])

    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(num_frame)):
        ret, frame = cap.read()

        if not ret:
            break

        # get boundding box by darknet
        resp = requests.post(
            'http://localhost:8000/image/darknet/detect/',
            files={'image_bytes': StringIO.StringIO(frame.tobytes())}
        ).json().get('bounding_box')

        if resp:
            for _, prob, box in resp:
                box = map(lambda x: int(x), box)
                box = [max(box[0] - box[2] / 2, 0),
                       max(box[1] - box[3] / 2, 0),
                       min(box[0] + box[2] / 2, 640),
                       min(box[1] + box[3] / 2, 480)]
                cropped_im = crop_and_resize(frame.copy(), box)
                # recognize
                resp = requests.post(
                    'http://localhost:8000/image/recognize/',
                    files={'image_bytes': StringIO.StringIO(cropped_im.tobytes())}
                ).content
                if resp != 'no face detected.':
                    resp = resp.split(',')
                    name = resp[0]
                    bb_bias = map(lambda x: float(x), resp[1:])
                    box[0] = int(box[0] + bb_bias[0] / 160 * (box[2] - box[0]))
                    box[1] = int(box[1] + bb_bias[1] / 160 * (box[3] - box[1]))
                    box[2] = int(box[2] - ((box[2] - box[0]) - bb_bias[2] / 160 * (box[2] - box[0])))
                    box[3] = int(box[3] - ((box[3] - box[1]) - bb_bias[3] / 160 * (box[3] - box[1])))
                    color = (0, 0, 255) if name == 'unknown' else (0, 255, 0)
                    im = Image.fromarray(frame)
                    drawer = ImageDraw.Draw(im)
                    drawer.rectangle(box, outline=color)
                    if name != 'unknown':
                        drawer.text(
                            (max(box[0], 10), max(box[1] - 30, 10)),
                            unicode(name, 'utf-8'),
                            fill=color,
                            font=ttfont
                        )
                    frame = np.array(im)

        out.write(frame)

    out.release()
    cap.release()


def play(fn_src):
    cap = cv2.VideoCapture(fn_src)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('player', frame)
        if cv2.waitKey(1000 / 25) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('func', default='main', help='function')
    parser.add_argument('fn_src', default='', help='fn_src')
    parser.add_argument('--fn_dst', default='', help='fn_dst')
    args = parser.parse_args()

    if args.func == 'capture':
        assert args.fn_dst
        capture(args.fn_dst)
    elif args.func == 'convert':
        assert args.fn_src
        assert args.fn_dst
        convert_size(args.fn_src, args.fn_dst)
    elif args.func == 'main':
        assert args.fn_src
        assert args.fn_dst
        main(args.fn_src, args.fn_dst)
    elif args.func == 'play':
        assert args.fn_src
        play(args.fn_src)
