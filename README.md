# FaceNetServer
Implement YOLOv2+MTCNN+FaceNet on Django Backend.

## Requirement

* python 2.7.10
* django 1.8.4
* tensorflow 1.4.1
* opencv-python 3.4.0
* pillow
* numpy
* pandas
* six
* tqdm

## Dataset

* CelebA for YOLOv2
* LFW for MTCNN and FaceNet

## Usage

* Run django server and activate model, including YOLOv2 model, MTCNN model and FaceNet model:

```
➜ cd FaceNetServer/
➜ python manage.py runserver 8000
➜ curl http://localhost:8000/init/
```

The activation process may take some minutes, just be patient.

* Add new member. Place your jpeg file (such as person.jpg) in FaceNetServer/client/images/, and execute commands as following:

```
➜ cd FaceNetServer/client/
➜ python upload_new.py images/person.jpg name
```

This would append embedding vector to FaceNetServer/facenet/facedb.csv and name to FaceNetServer/facenet/record.txt, at the same time generate an image FaceNetServer/client/test.jpg, take a look to ensure the MTCNN selects an ideal box.

* Recognize. Limited by the server network, we handle video files only, you can modified main function in FaceNetServer/client/video.py, specify the input of opencv capture to usb camera, and remember to set the frame size as (640, 480). Assume we have a video file FaceNetServer/client/test.mp4, excutes commands as following:

```
➜ cd FaceNetServer/client/
➜ python video.py convert test.mp4 resized_video.mp4
➜ python video.py main resized_video.mp4 recognized_video.mp4
```

command "convert" resize test.mp4 to size (640, 480), and save the resized video to resized_video.mp4. You can search the project and modify all (640, 480) to other (width, height), if size not match, recognizition result would be wrong.

command "main" read frames from resized\_video.mp4, selecting box using YOLOv2, fine-tunning box using MTCNN, and recognize using FaceNet, the boxes and recognized names would be plotted on frames and save to recognized\_video.mp4.

You can specify recognize-threshold in file FaceNetServer/facenet/thresh.txt.

**NOTE: Any modified on django files would result in restart of django, try not to modified views.py, urls.py, detector.py, ops.py after django initialized, or you should initialize django again, and this take much time.**

## Running on GPU

* Ensure your TensorFlow is compiled on CUDA/CUDNN.
* Ensure your libdarknet.so is make on your own machine. You can download source code of [darknet](https://github.com/pjreddie/darknet), edit GPU=1 in Makefile and execute "make", this would generate libdarknet.so, and replace FaceNetServer/client/libdarknet.so and FaceNetServer/facenet/darknet/libdarknet.so.

## Other Files

* [Video samples](https://pan.baidu.com/s/15LfRbmbdTXvLjD7aCBwQ9A)
* [Source code and model parameters](https://pan.baidu.com/s/1m-0WMAtVCTmlLDrnKiXMrQ)

## Reference

* You Only Look Once: Unified, Real-Time Object Detection.
* Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.
* FaceNet: A Unified Embedding for Face Recognition and Clustering.

## Collaborator

* Chenquan Huang