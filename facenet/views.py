import os
import codecs
import linecache
import numpy as np
import pandas as pd
import tensorflow as tf
import ops
import detector
from PIL import Image
from pandas.errors import EmptyDataError
from types import NoneType

from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST, require_GET

# params of detector
minsize = 50  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 0

# path of facedb and person name record
facedb_path = os.path.join('facenet', 'facedb.csv')
record_path = os.path.join('facenet', 'record.txt')

print
print 'Loading Face Embedding Model ...'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
if os.path.exists(os.path.join('facenet', 'ckpt')):
    meta_file = '20170512-110547/model-20170512-110547.meta'
    ckpt_file = '20170512-110547/model-20170512-110547.ckpt-250000'
    # meta_file = '20180402-114759/model-20180402-114759.meta'
    # ckpt_file = '20180402-114759/model-20180402-114759.ckpt-275'
    # meta_file = '20180408-102900/model-20180408-102900.meta'
    # ckpt_file = '20180408-102900/model-20180408-102900.ckpt-90'
    saver = tf.train.import_meta_graph(os.path.join('facenet', 'ckpt', meta_file))
    saver.restore(sess, os.path.join('facenet', 'ckpt', ckpt_file))
inputs = sess.graph.get_tensor_by_name("input:0")
outputs = sess.graph.get_tensor_by_name("embeddings:0")
phase_train = sess.graph.get_tensor_by_name("phase_train:0")

print
print 'Loading MTCNN Model ...'
mtcnn_sess = tf.Session()
pnet, rnet, onet = detector.create_mtcnn(mtcnn_sess)

from darknet.detector import detect as dn_detect


@require_GET
def init_sys(requests):
    return HttpResponse('initial success.')


@require_POST
def recognize(requests):
    """
    :param requests: Image.
    :return: Name of the person in received image.
    """
    im = ops.read_bytes(requests, (160, 160), reverse_rgb=True)

    im_and_bb = detect(im)

    if type(im_and_bb) == NoneType:
        return HttpResponse('no face detected.')

    im, bb = im_and_bb

    embedding = sess.run(outputs, {inputs: im, phase_train: False})

    facedb = pd.read_csv(facedb_path, header=None).values

    similarity = np.sum(np.square(np.subtract(facedb, embedding)), axis=1)
    print similarity

    with open(os.path.join(os.path.dirname(__file__), 'thresh.txt'), 'r') as fp:
        thresh = float(fp.read().strip())

    bb = map(lambda x: str(x), bb)
    if np.min(similarity) > thresh:
        return HttpResponse('unknown' + ',' + ','.join(bb))
    else:
        reload(linecache)
        line = linecache.getline(record_path, np.argmin(similarity) + 1).strip()
        print 'recognize:', line
        return HttpResponse(line + ',' + ','.join(bb))


@require_POST
def new(requests):
    """
    :param requests: Image, Name
    :return: None
    :description: Add new portraits to facedb
    """
    im = ops.read_and_resize_image(requests)

    im_and_bb = detect(im)

    if type(im) == NoneType:
        return HttpResponse('no face detected.')

    im, bb = im_and_bb

    embedding = sess.run(outputs, {inputs: im, phase_train: False})

    try:
        facedb = pd.read_csv(facedb_path, header=None).values
        facedb = np.concatenate([facedb, embedding], axis=0)
    except EmptyDataError:
        facedb = embedding

    pd.DataFrame(facedb).to_csv(facedb_path, index=False, header=None)

    with codecs.open(record_path, 'a', encoding='utf-8') as fp:
        name = requests.POST.get('name')
        fp.write(name + '\n')

    return HttpResponse('success')


def detect(im):
    bounding_boxes, _ = detector.detect_face(im, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        return None

    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, im.shape[1])
    bb[3] = np.minimum(det[3] + margin / 2, im.shape[0])
    cropped = im[bb[1]:bb[3], bb[0]:bb[2], :]
    im = Image.fromarray(cropped, 'RGB')
    im = im.resize((160, 160))
    im = ops.normalize_image(im)
    return im, bb


@require_POST
def mark(requests):
    """
    :param requests: Image
    :return: Cropped face
    """
    im = ops.read_image(requests)

    bounding_boxes, _ = detector.detect_face(im, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        return HttpResponse('no face detected.')

    det = np.squeeze(bounding_boxes[0, 0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, im.shape[1])
    bb[3] = np.minimum(det[3] + margin / 2, im.shape[0])

    return HttpResponse(' '.join(map(lambda i: str(i), list(bb))))


@require_POST
def darknet_detect(requests):
    """
    :param requests: Image
    :return: [(prob, (bounding_boxes))]
    """
    im = ops.read_bytes(requests, (640, 480))
    return JsonResponse({'bounding_box': dn_detect(im)})
