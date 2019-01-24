import numpy as np
import cv2
 
cap = cv2.VideoCapture('videoplayback.avi')

k = cv2.waitKey(1) & 0xFF
print(ord('q'))

while(True):

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
  elif ret == False:
  	

  cv2.cvtColor()
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()