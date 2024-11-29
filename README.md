[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/qmA6zave)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17373523&assignment_repo_type=AssignmentRepo)
# HOMEWORK8
HOMEWORK 8

TODO:
1) Print the image (or hard copies will be given in class) and fill out the "your name", the 1390 vs 2390 checkbox, and enter a random 5 digit number

2) Take a photograph of the paper

3) Using the ACUCO symbols, align and orient the photograph 

4) Using EasyOCR, read in the name box that you wrote

5) Write code to check which of the two checkboxes is marked by comparing the intensity in the two boxes

6) Using the example from class on OCR-digits, decode the 5 numbers you wrote

6) Using Tesseract-OCR, read the printed text box ("some test to decode by openCV')

ECE2390 students
7) Using MediaPipe, find the facial features for the image of Dr Huppert and draw the markers on the image.

Note, MediaPipe only works in Python3.11 or lower (the newest 3.12 or 3.13beta have issues with the wheel/Pip install).

---------------------------

Here are the corrdinates that can be used for the masks for the regions 
All positions are in cm.  Note, the horizontal axis (x) is defines such that right is positive.  The vertical axis (Y) is defined such that up is negative and down is positive.

Red dot
[H,W] = 0.5cm
[X,Y] = [0cm , 0cm]

TEXT BOX.  
[H,W] = 2cm x 10cm 
[X,Y] = [-10cm, -13cm]

ACURO #1 (Upper Right)
[H,W] = 4cm x 4cm	
[X,Y] = [6cm, -13cm]

ACURO #2 (Upper Right)
[H,W] = 4cm x 4cm	
[X,Y] = [-10cm, 9cm]

ACURO #3 (Lower Right)
[H,W] = 4cm x 4cm	
[X,Y] = [6cm, 9cm]

Checkbox ECE-1390
[H,W] = 1cm x 2cm	
[X,Y] = [-10cm, -9cm]

Checkbox ECE-2390
[H,W] = 1cm x 2cm	
[X,Y] = [-10cm, -7cm]

5-digit number
[H,W] = 1.5cm x 1.5cm	
[X,Y] = [{-9,-7.5,-6.-4.5,-3} cm,-4cm]

Some Text to decode
[H,W] = 5cm x 12cm	
[X,Y] = [-10cm, 1cm]

Face Picture
[H,W] = 10cm x 7cm	
[X,Y] = [3cm, -3cm]


