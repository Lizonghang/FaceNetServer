import darknet

darknet.set_gpu(0)
net = darknet.load_net("cfg/yolo-voc.2.0.cfg", "yolo-voc_30000.weights", 0)
meta = darknet.load_meta("cfg/voc.data")


def detect(im):
    """im: opencv format image data"""
    im = darknet.array_to_image(im)
    darknet.rgbgr_image(im)
    return darknet.detect(net, meta, im)


if __name__ == '__main__':
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from datetime import datetime

    im = Image.open('data/photo.jpg')
    out = im.copy()
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # im = cv2.imread('data/photo.jpg')
    bbs = detect(im)

    drawer = ImageDraw.Draw(out)
    ttfont = ImageFont.truetype('simsun.ttc', 20)
    for _, prob, box in bbs:
        box = map(lambda x: int(x), box)
        box = [max(box[0] - box[2] / 2, 0),
               max(box[1] - box[3] / 2, 0),
               min(box[0] + box[2] / 2, 640),
               min(box[1] + box[3] / 2, 480)]
        drawer.rectangle(box, outline=(0, 0, 255))
        drawer.text(
            (10, 10),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            fill=(0, 0, 255),
            font=ttfont
        )
    out.show()
