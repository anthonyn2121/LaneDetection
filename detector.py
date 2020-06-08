import numpy as np 
import cv2  
import math 


class LaneDetector(object): 
    def __init__(self, image): 
        self.image = image 
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS) 

    def adjust_gamma(self, image, gamma=1.0):
        '''
        Darkens a grayscale image by adjusting gamma value 
        Arguments: 
            image: single channel grayscale image 
            gamma: gamma value to change  
        Returns: 
            the adjusted image 
        '''
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def isolate_white_yellow(self, image):
        '''
        Get color masks between threshold value 
        Arguments: 
            image: HLS image 
        Returns: 
            an image; Bitwise OR of yellow and white masks
        '''
        # isolate yellow from HLS color space 
        yllw_lower = np.array([10,0,100], dtype=np.uint8) 
        yllw_upper = np.array([40,255,255], dtype=np.uint8)
        yllw_mask = cv2.inRange(image, yllw_lower, yllw_upper)
        
        # isolate white from HLS color space 
        wht_lower = np.array([0,200,0], dtype=np.uint8) 
        wht_upper = np.array([200,255,255], dtype=np.uint8)
        wht_mask = cv2.inRange(image, wht_lower, wht_upper)

        combined = cv2.bitwise_or(wht_mask, yllw_mask)
        return combined 
    

    def detect_edges(self, image, kernel_size=5): 
        ''' 
        Apply gaussian blur and then apply canny edge detection 
        Arguments: 
            image: applied bitwise_or image 
            kernel_size: (int) size of kernel 
        Returns: 
            canny: Applied image 
        ''' 
        # apply gaussian blur 
        blur = cv2.GaussianBlur(self.gray, ksize=(kernel_size, kernel_size), sigmaX=0)
        # canny edge detector 
        canny = cv2.Canny(blur, 70, 150)
        return canny 


    def get_roi(self, image):
        '''
        Returns mask that only displays region of interest 
        Arguments: 
            image: Canny applied image
        Returns: 
            Masked image 
        '''
        rows, cols = image.shape[:2]
        mask = np.zeros_like(image)

        bottom_left = [cols*0.1, rows]
        bottom_right = [cols*0.95, rows]
        top_left = [cols * 0.4, rows * 0.61]
        top_right = [cols * 0.6, rows * 0.61]

        vertices = np.array([
            [bottom_left, top_left, top_right, bottom_right]
        ], dtype=np.int32)

        if len(mask.shape) == 2: 
            cv2.fillPoly(mask, vertices, 255)
        else: 
            cv2.fillPoly(mask, vertices, (255,) * mask.shape[2]) 
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image, rho=1, theta=np.pi/180, threshold=20, minLineLen=20, maxLineGap=300): 
        '''
        Apply hough transformation 
        ''' 
        lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), 
            minLineLength=minLineLen, maxLineGap=maxLineGap)
        
        return lines
    
    def draw_lines(self, image, lines): 
        for line in lines: 
            for x1, y1, x2, y2 in line: 
                cv2.line(image, (x1, y1), (x2, y2), color=[0,0,255], thickness=5)
        return image 


    def main(self, image): 
        gray = self.adjust_gamma(self.gray, 0.5)
        combined_mask = self.isolate_white_yellow(self.hls)
        canny = self.detect_edges(combined_mask, kernel_size=7)
        roi = self.get_roi(canny)
        lines = self.hough_transform(roi)
        lanes = self.draw_lines(image, lines)
        return combined_mask, lanes 



if __name__ == "__main__": 
    # path = '/home/anthony/Documents/lane_detection/solidWhiteRight.mp4'
    path = '/media/anthony/My Book/Khang School/Independent Projects/lane_detection/solidWhiteRight.mp4'
    cap = cv2.VideoCapture(path)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (960, 540))
    while (cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True:
            # Pipeline here 
            detector = LaneDetector(frame) 
            mask, output = detector.main(frame)
            cv2.imshow('Color Isolation', mask)
            cv2.imshow('Output', output)  
            # write to video file 
            out.write(output)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break 
        else: 
            break 
    cap.release()
    out.release()
    cv2.destroyAllWindows()