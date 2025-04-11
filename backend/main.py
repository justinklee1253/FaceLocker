import cv2 as cv #import opencv for computer vision 
import sys
import matplotlib.pyplot as plt


def detect_bounding_box(vid): 
    """
    This function converts images/video stream to gray scale to simplify them.

    Optimize computational efficiency, simplify algorithms, improve feature and edge detection. 

    Use pre-loaded face classifier (haar cascade) to detect faces

    Then we want to scale the image slightly larger to make the detection of the face easier.

    Must be 5 neighboring windows (rectangles) for the face to be valid. 

    Draws a green rectangle around each detected face in the frame
    """
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY) #convert color image to grayscale image (image to be converted, color conversion code)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


#UNCOMMENT FOR IMAGE READING 

# imagePath = 'propic.png' 
# img = cv.imread(imagePath) #read's image: load's image from the file path and return it in the form of a numpy array 
# img.shape #contain's dimensions of image: (height, width, number of color channels in image)
# print(img.shape) #debugging

# gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
# gray_image.shape #calls shape attribute of image 
# print(gray_image.shape) 

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
#CascadeClassifier is a class in OpenCV for object detection, haarcascade = pretrained classifier to detect faces in image or video stream

video_capture = cv.VideoCapture(0) #0 tells opencv to use default camera on device.
print(f"OpenCV version: {cv.__version__}")

if not video_capture.isOpened:
    print("not opened")
    exit()

while True:
    result, video_frame = video_capture.read() #read frames from video 
    #read returns 2 values: result (A boolean which indicates if frame was successfully read), video_frame: actual image data of the captured frame
    if result is False:
        break

    faces = detect_bounding_box(video_frame)

    cv.imshow( #display image in a separate window
        "Face Detection Prototype", video_frame
    ) #display's processed frame in a window named "Face Detection Prototype"

    if cv.waitKey(1) & 0xFF == ord("q"): #wait 1 ms for keyboard input, allowing OpenCV to process GUI 
        #if q is pressed, loop breaks and program ends
        break

video_capture.release() #releases video resource
cv.destroyAllWindows() #closes all opencv windows



# for (x, y, w, h) in face: #draws a green rectangle around each detected face
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# # x, y axis in which the faces were detected and their width and height 
# # (0, 255, 0) represents color of bounding box (green), 4 is the thickness
# # w is the width of the rectangle, and h height is the rectangle 

# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) #convert image from BGR to RGB 

# #Plot the figure using matplotlib
# plt.figure(figsize=(20,10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()