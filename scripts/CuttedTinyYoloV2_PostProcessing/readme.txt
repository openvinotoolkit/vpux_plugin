This directory contains 5 files:
- readme.txt - this file
- main.py - the script to run to obtain results;
- aug.py, utils.py - two files with auxiliary functions and classes;
- tiny-yolov2-128output.tflite - cutted version of Tiny YOLO v2 model (without region detection layer).

Usage of script:

python3 main.py <imageFileName> <probabilityForObjectDetection>

or

python3 main.py <imageFileName>

In the last case when you do not set probabilityForObjectDetection explicity then its value will be set to 0.4.



Example of usage and some results:

1. python3 main.py cat.jpg 0.4

	Bounding boxes:
	Bounding box #0
	Object : cat
	Probability = 0.5801697128383008
	Coordinates (x_min, y_min - x_max, y_max) : 145, 3 - 390, 359


2. python3 main.py person.bmp 0.3

	Bounding boxes:
	Bounding box #0
	Object : person
	Probability = 0.3367210947408788
	Coordinates (x_min, y_min - x_max, y_max) : 107, 96 - 185, 373

	Bounding box #1
	Object : cow
	Probability = 0.40541659425886273
	Coordinates (x_min, y_min - x_max, y_max) : 276, 138 - 385, 336

	Bounding box #2
	Object : dog
	Probability = 0.3750763282986678
	Coordinates (x_min, y_min - x_max, y_max) : 50, 256 - 128, 357

3. python3 main.py dog_bicycle_car.bmp 0.3

	Bounding boxes:
	Bounding box #0
	Object : car
	Probability = 0.41645595391229157
	Coordinates (x_min, y_min - x_max, y_max) : 264, 65 - 415, 136

	Bounding box #1
	Object : bicycle
	Probability = 0.33726744707925643
	Coordinates (x_min, y_min - x_max, y_max) : 54, 90 - 356, 319

	Bounding box #2
	Object : dog
	Probability = 0.32101479962268487
	Coordinates (x_min, y_min - x_max, y_max) : 36, 142 - 187, 401
