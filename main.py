import cv2
import os
import numpy as np
from scipy.signal import correlate2d
class ConvolutionSystem:

    def __init__(self, inputImage, templateImage):
        self.image = cv2.imread(inputImage)
        self.kernel = cv2.imread(templateImage)
        self.myConvFound = 0
        self.convFound = 0
        self.correlationFound = 0

    
    def convolution(self):
        image = self.image.copy()
        
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_kernel = cv2.cvtColor(self.kernel, cv2.COLOR_BGR2GRAY)

        gray_image = np.float32(gray_image)
        gray_kernel = np.float32(gray_kernel)

        # Convolution using the OpenCV function
        result = cv2.matchTemplate(gray_image - gray_image.mean(), gray_kernel - gray_kernel.mean(), cv2.TM_CCOEFF_NORMED)

        # Define a threshold for matches
        threshold = 0.61
        self.convFound = 0

        #display the Feature Map
        cv2.imshow('features', result)
        cv2.waitKey(0)


        # Iterate through all matches above the threshold
        while True:
            # Find the maximum value in the result array
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Check if the maximum value is above the threshold
            if (max_val >= threshold):
                # Get the coordinates of the matched area
                top_left = max_loc
                bottom_right = (top_left[0] + self.kernel.shape[1], top_left[1] + self.kernel.shape[0])

                # Draw a rectangle around the matched region
                cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

                # Zero out the current match circle
                cv2.circle(result, center=max_loc, radius=60, color=(0, 0, 0), thickness=-1)

                # Increment the number of found templates
                self.convFound += 1

            else:
                break
        
        # Print the number of found templates
        print('Found {} templates'.format(self.convFound))
        # Display the result image
        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def correlation(self):
        image = self.image.copy()

        image1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(self.kernel, cv2.COLOR_BGR2GRAY)

        corr = correlate2d(image1, image2 - image2.mean(), mode='valid')
        corr = (corr - np.min(corr)) / (np.max(corr) - np.min(corr)) * 255
        corr = np.uint8(corr)

        threshold = 225
        self.correlationFound = 0
        cv2.imshow('features', corr)
        cv2.waitKey(0)


        while True:

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)

            if (max_val >= threshold):

                top_left = max_loc
                bottom_right = (top_left[0] + self.kernel.shape[1], top_left[1] + self.kernel.shape[0])
                cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)
                cv2.circle(corr, center=max_loc, radius=60, color=(0, 0, 0), thickness=-1)

                self.correlationFound += 1

            else:
                break

        print('Found {} templates'.format(self.correlationFound))

        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def myConv(self):
        image = self.image.copy()
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.kernel, cv2.COLOR_BGR2GRAY)


        result = self.__convol2D(gray1, gray2)

        threshold = 0.80
        self.myConvFound = 0

        cv2.imshow('features', result)
        cv2.waitKey(0)

        while True:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if (max_val >= threshold):
                top_left = (max_loc[0] - int(self.kernel.shape[1] / 2), max_loc[1] - int(self.kernel.shape[0] / 2))
                bottom_right = (top_left[0] + self.kernel.shape[1], top_left[1] + self.kernel.shape[0])

                cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)
                cv2.circle(result, center=max_loc, radius=60, color=(0, 0, 0), thickness=-1)

                self.myConvFound += 1

            else:
                break

        print('Found {} templates'.format(self.myConvFound))

        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
            
    def __convol2D(self, image, kernel):

        kernel = (kernel.astype("float32") / 255.0) * 2 - 1
        kernel /= kernel.sum()
        
        image_height, image_width = image.shape[:2]
        kernel_height, kernel_width = kernel.shape[:2]

        buffHeight = int(kernel_height / 2)
        buffWidth = int(kernel_width / 2)

        output = np.zeros((image_height + buffHeight * 2, image_width + buffWidth * 2), dtype="float32")
        output[buffHeight:-buffHeight, buffWidth:-buffWidth] = image.astype(np.float32) / 255.0

        result = np.zeros((image_height, image_width), dtype="float32")

        for y in np.arange(0, image_height):
            for x in np.arange(0, image_width):

                roi = output[y : y + kernel_height, x : x + kernel_width]
                result[y, x] = (np.multiply(roi, kernel)).sum()

        return result



if __name__ == '__main__':
    
    data_path = 'Data'
    temp_data_path = 'Data/temp_data_10.PNG'
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(data_path, filename)
            app = ConvolutionSystem(file_path, temp_data_path)
            print("Picture: ", filename)
            print("----Ready Convolution Function----")
            app.convolution()
            print("----My Convolution Function----")
            app.myConv()
            print("----Ready Correlation Function----")
            app.correlation()

            