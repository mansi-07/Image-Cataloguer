# basic_yolo_webapp
This is the web part which helps you interact with the image cataloguer model

It takes in an image as the input and stores it in the static/input folder.
The model takes the input from the input folder and outputs the images in the static/pics folder.
All the images present in the output folder are then dispalyed on the web page.

Before running the yolo.py the files feature_list.csv,coco.names, yolov4.cfg ,yolov4.weights must be present in the folder.(same level as yolo.py)

The cataloguer dataset must also be present in the folder.

The notebook has also been included.(image cataloguer3).

Commands to run the app-
open the terminal and cd into this folder and type-


$env:FLASK_APP='yolo.py'


flask run

To turn on debug mode type-


$env:FLASK_APP='yolo.py'


$env:FLASK_DEBUG=1


flask run
