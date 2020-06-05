# Highway Lane Detection System 

A personal project of mine, and also the beginning to my long journey of studying self-driving vehicle technologies. The method used in this project is the [hough transform](http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf). The video that I test my method on is from the [Udacity Self-driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) so I could compare my output with the students of that course. 

The pipeline that I followed came from [this blog](https://towardsdatascience.com/finding-lane-lines-simple-pipeline-for-lane-detection-d02b62e7572b). To summarize: 

```
1. Convert original image to grayscale 
2. Convert orginal image to HLS color space 
3. Isolate yellow and white colors to create a mask 
4. Bitwise-AND mask with grayscale image 
5. Apply gaussian blur 
6. Apply Canny edge detector 
7. Get Region of interest 
8. Get Hough lines 
9. Consolidate Hough lines and draw them on the original image 
```

## Getting Started 
These instructions will get a copy of the project up and running on your local machine 
### Requirements
- Python 3.6 
- numpy 
- cv2 
### Usage 
``` 
python3 detector.py 
``` 
