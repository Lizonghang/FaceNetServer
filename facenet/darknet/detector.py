import darknet
import os

print
print 'Loading YOLOv2 Model ...'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# darknet.set_gpu(0)
net = darknet.load_net(
    os.path.join(os.path.dirname(__file__), "cfg/yolo-voc.2.0.cfg"),
    os.path.join(os.path.dirname(__file__), "yolo-voc_30000.weights"),
    0)
meta = darknet.load_meta(os.path.join(os.path.dirname(__file__), "cfg/voc.data"))


def detect(im):
    """im: opencv format image data"""
    im = darknet.array_to_image(im)
    darknet.rgbgr_image(im)
    return darknet.detect(net, meta, im)
