
import cv2
import os

# define a video capture object
vid = cv2.VideoCapture(0)
result = True
while (result):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("NewPicture.jpg", frame)
        
        result = False
        
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()