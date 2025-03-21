import cv2 as cv
import numpy as np

class ExtendeDoGFilter:
    def __init__(self, image: str, sigma: float = 1, k: float = 1., p: float =1., phi: float = 0.1, e: float = 0.1) -> None:
        self.sigma = sigma
        self.k = k
        self.p = p
        self.phi = phi
        self.e = e
        self.image_array = cv.imread(image)
        assert self.image_array is not None, "Image not found"
        self.image_array = self.image_array.astype(np.float32) / 255.0

    def set_image_array(self, pathname: str) -> np.ndarray:
        """
        Input: pathname to the image_array.\n
        Set the image_array to a new image_array if needed.
        """
        #read the image_array as a grayscale image_array
        self.image_array = cv.imread(pathname)
        assert self.image_array is not None, "Image not found"
        self.image_array = self.image_array.astype(np.float32) / 255.0

    def dog(self) -> np.ndarray:
        """
        Input: image_array as a numpy array.\n
        Output: Difference of Gaussians applied on a numpy array.\n
        Computes the reparametrized Difference of Gaussians on a grayscale image array
        """ 
        gray_image = cv.cvtColor(self.image_array, cv.COLOR_BGR2GRAY)
        #set trackbar sliders values
        # sigma is the standard deviation value of the blurs, 0.01 is the minimum value of sigma to avoid ksize = 0
        self.sigma = max(cv.getTrackbarPos("Sigma", "XDoG") / 100, 0.01)
        #k is the scaling factor of the second gaussian blur, 0.01 for the same reason as sigma
        self.k = max(cv.getTrackbarPos("K", "XDoG") / 100, 0.01)
        #p will be the weight of the two blurred image_arrays, it acts like a sharpness parameter
        self.p = cv.getTrackbarPos("P", "XDoG") / 100

        #compute the two gaussian blurs, one of them scaled by k
        blur1 = cv.GaussianBlur(gray_image, (0, 0), self.sigma)
        blur2 = cv.GaussianBlur(gray_image, (0, 0), self.sigma * self.k)

        #return the parametrized difference of the two blurred image_arrays
        return (1 - self.p) * blur1 + self.p * blur2

    def threshold(self, dog_array: np.ndarray, masking: bool = False) -> np.ndarray:
        """
        Input: DoG applied image array as a numpy ndarray.\n
        Output: Thresholded image array as a numpy ndarray.\n
        Computes the thresholding of the image arrray using a function.\n
        If masking is True, the thresholded image_array is multiplied by the original image as a grayscale mask
        """
        #set trackbar values
        #phi is the sharpness of the ramp function, it increases the contrast of the image
        self.phi = cv.getTrackbarPos("Phi", "XDoG") / 100
        #epsilon is the threshold of the ramp function, it is the minimum value of the image to be displayed
        self.e = cv.getTrackbarPos("Epsilon", "XDoG") / 100

        #use different functions for the thresholding, e.g. sigmoid, ReLU, exponential, etc. Be creative!
        result = np.where(dog_array >= self.e, 1, 1 + np.tanh(self.phi * (dog_array - self.e)))
        
        if masking:

            original = self.image_array
            #create a 3d thresholded ndarray to multiply by the original image_array
            thresholded_3d = result[:, :, np.newaxis]
            #multiply the thresholded image_array by the original image_array to get the final result
            result = original * thresholded_3d
            result = (result * 255).astype(np.uint8)
        
        #normalize the array to 0-255
        result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX)
        
        return result

    def set_slider_value(self, value: float) -> float:
        """
        Input: value as a float.\n
        Output: value as a float.\n
        Sets the slider value to the input value
        """
        return value

    def create_trackbars(self) -> None:
        """
        Creates the trackbars for the XDoG filter
        """
        cv.namedWindow("XDoG")
        cv.createTrackbar("Sigma", "XDoG", 1, 5000, self.set_slider_value)
        cv.createTrackbar("K", "XDoG", 1, 5000, self.set_slider_value)
        cv.createTrackbar("P", "XDoG", 1, 5000, self.set_slider_value)
        cv.createTrackbar("Phi", "XDoG", 1, 5000, self.set_slider_value)
        cv.createTrackbar("Epsilon", "XDoG", 1, 5000, self.set_slider_value)
