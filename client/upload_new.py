import requests
import argparse
import os
from PIL import Image, ImageDraw


def main(fn, name):
    files = {'image': open(fn, 'rb').read()}

    # new person
    url = 'http://localhost:8000/image/new/'
    os.system('curl -F image=@{fn} -F name={name} {url}'.format(fn=fn, name=name, url=url))

    # get bounding box
    url = 'http://localhost:8000/image/mark/'
    resp = requests.post(url, files=files).content
    if resp == 'no face detected.':  return resp
    box = map(lambda item: float(item), resp.split())

    # plot box on image
    im = Image.open(fn)
    color = (255, 0, 0)
    drawer = ImageDraw.Draw(im)
    drawer.rectangle(box, outline=color)
    im.save('test.jpg')

    print
    print 'Save recognized image to /client/test.jpg'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', help='fn')
    parser.add_argument('name', help='name')
    args = parser.parse_args()
    main(args.fn, args.name)
