from dog import ExtendeDoGFilter
import cv2 as cv


if __name__ == "__main__":
    """
    Main function to run the Difference of Gaussians filter on an image.
    """
    #initialize the filter with the image path
    dog_filter = ExtendeDoGFilter("CG/Difference_of_Gaussians/examples/pond.png")
    dog_filter.create_trackbars()
    
    while True:
        #get the DoG applied image
        dog_image = dog_filter.dog()
        #get the thresholded image
        thresholded_image = dog_filter.threshold(dog_image, masking=True)
        #display the image
        cv.imshow("XDoG", thresholded_image)
        #wait for the user to press a key
        key = cv.waitKey(1) & 0xFF
        #if the user presses 'q', break the loop
        if key == ord("q"):
            break
        #if the user presses 's', save the image
        if key == ord("s"):
            cv.imwrite("CG/Difference_of_Gaussians/examples/pond_xdog.png", thresholded_image)
            break
    cv.destroyAllWindows()