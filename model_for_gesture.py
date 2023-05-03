import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = keras.models.load_model(r"CNNmodel.h5")

bg = None
acc_weight = 0.5

region_top = 100
region_bottom = 300
region_right = 150
region_left = 350

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J'}

def accumulateAvg(frame, acc_weight):

    global bg
    
    if bg is None:
        bg = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, bg, acc_weight)



def segmentHand(frame, threshold=25):
    global bg
    
    diff = cv2.absdiff(bg.astype("uint8"), frame)

    
    _ , thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Keep track of the image's external contours
    img, contour, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour) == 0:
        return None
    else:
        # The biggest external contour should be the hand...
        max_contour = max(contour, key=cv2.contourArea)
 
        return (thresh, max_contour)

cam = cv2.VideoCapture(0)
frame_count =0
while True:
    ret, frame = cam.read()

    # we flip the frame in order to avoid inverted image of captured frame
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[region_top:region_bottom, region_right:region_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    # configuring hand into the ROI
    if frame_count < 70:
        
        accumulateAvg(gray_frame, acc_weight)
        
        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2)
    
    else: 
        # segmenting the hand region
        hand = segmentHand(gray_frame)
        

        # Checking if hand is detected
        if hand is not None:
            
            thresh, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (region_right, region_top)], -1, (255, 0, 0),1)
            
            cv2.imshow("Thesholded Hand Image", thresh)
            
            thresh = cv2.resize(thresh, (64, 64))
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            thresh = np.reshape(thresh, (1,thresh.shape[0],thresh.shape[1],3))
            
            pred = model.predict(thresh)
            cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (region_left, region_top), (region_right, region_bottom), (255,128,0), 3)

    # incrementing no:of frames for tracking
    frame_count += 1

    # display frame with segmented hand
    cv2.putText(frame_copy, "Sign-Language-Recognition", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (51,255,51), 1)
    cv2.imshow("Sign Detection", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera & destroy all the windows...
cam.release()
cv2.destroyAllWindows()
