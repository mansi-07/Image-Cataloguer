# Image Cataloguer
This project was done under the **Crypt** Special Interest Group of **ISTE-NITK**. 
This is the web interface which helps you interact with the image cataloguer model.

The web interface takes in an image as the input and stores it in the static/input folder.
The deep learning model takes the input from the input folder and outputs the images in the static/pics folder.
All the images present in the output folder are then dispalyed on the web page.

Parallelly, the Yolov4 model prints out the elements present in the images(Elements are selected from a class of 80 objects).

Also the dominant colors present in the image are displayed as tags along with the element tags.

Before running the yolo.py the files feature_list.pkl, coco.names, yolov4.cfg, yolov4.weights must be present in the folder.(same level as yolo.py)

The cataloguer dataset must be present in the **static folder**.(in the same level as inputs/ and pics/).

The notebook has also been included(image cataloguer3 and colors).

>Commands to run the app-
open the terminal and cd into this folder and type-

```
$env:FLASK_APP='yolo.py'
flask run
```
>To turn on debug mode type-

```
$env:FLASK_APP='yolo.py'
$env:FLASK_DEBUG=1
flask run
```


**Dataset**: https://www.kaggle.com/rahulxd/image-cataloguer/

**Link for COCO.Names used**: https://drive.google.com/file/d/1YWagPmrNetdgydR3z7rqr_d9ABvkboQW/view?usp=sharing

**Reverse Image Search Model**: https://www.kaggle.com/salonimathur16/notebook86698da377

**YoloV4 Model**: https://www.kaggle.com/aneesh2002/yolov4-project

**YoloV4 Weights**: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

**YoloV4 Config**: https://github.com/kiyoshiiriemon/yolov4_darknet/blob/master/cfg/yolov4.cfg
