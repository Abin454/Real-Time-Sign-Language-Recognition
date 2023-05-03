import cv2
import numpy as np

bg = None 
acc_weight = 0.5 
#------------------------------------------------------------------
region_top = 101 
region_bottom = 301
region_right = 151
region_left = 351
#------------------------------------------------------------------
def accumulateAvg(fr, acc_weight):

    global bg
    if bg is None:
        bg = fr.copy().astype("float")
        return None

    cv2.accumulateWeighted(fr, bg, acc_weight)
#-------------------------------------------------------------------
def segmentHand(fr, threshold=25): 
    global bg
    
    diff = cv2.absdiff(bg.astype("uint8"), fr)

    _ , thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Keep track of the image's external contours
    img, contour, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour) == 0:
        return None
    else:
        
        max_contour = max(contour, key=cv2.contourArea)
        
        return (thresh, max_contour)
#--------------------------------------------------------------------
cam = cv2.VideoCapture(0)

frames_count = 0
letter = input("Letter to create gesture: ")
img_count = 0
flag = True
while flag == True:
    ret,fr = cam.read()

    # we flip the frame in order to avoid inverted image of captured frame
    fr = cv2.flip(fr, 1)

    frame_copy = fr.copy()

    region_of_intrest = fr[region_top:region_bottom, region_right:region_left]

    gray_frame = cv2.cvtColor(region_of_intrest, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if frames_count < 60:
        accumulateAvg(gray_frame, acc_weight)
        if frames_count <= 59:
            
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2)
            #cv2.imshow("Sign Detection",frame_copy)
         
    # configuring hand into the ROI
    elif frames_count <= 300: 

        hand = segmentHand(gray_frame)
        
        cv2.putText(frame_copy, "Adjust hand...Gesture for" + str(letter), (200, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
        
        # Counting the amount of contours observed to see if a hand has been detected
        if hand is not None:
            
            thresh, handSegment = hand

            # Make a contour drawing around the hand portion
            cv2.drawContours(frame_copy, [handSegment + (region_right, region_top)], -1, (255, 0, 0),1)
            
            cv2.putText(frame_copy, str(frames_count)+"For" + str(letter), (70, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)

            # Display the thresholded image
            cv2.imshow("Thresholded Hand Image", thresh)
    
    else: 
        
        # Segmenting the hand region
        hand = segmentHand(gray_frame)
        
        # Check if hand detected
        if hand is not None:
            
            # unpack the thresholded img and the max_contour
            thresh, handSegment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [handSegment + (region_right, region_top)], -1, (255, 0, 0),1)
            
            cv2.putText(frame_copy, str(frames_count), (70, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            #cv2.putText(frame_copy, str(frames_count)+"For" + str(letter), (70, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            cv2.putText(frame_copy, str(img_count) + 'images' +"For" + str(letter), (200, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            
            # Display the thresholded image
            cv2.imshow("Thresholded Hand Image", thresh)
            if img_count <= 300:
                if img_count<=39:
                    cv2.imwrite(r"C:\\gesture\\test\\"+str(letter)+"\\" + str(img_count) + '.jpg', thresh)
                cv2.imwrite(r"C:\\gesture\\train\\"+str(letter)+"\\" + str(img_count) + '.jpg', thresh)
                #cv2.imwrite(r"C:\\gesture\\train\\" + str(img_count) + '.jpg', thresh)
            else:
                flag = False
            img_count = img_count + 1
        else:
            cv2.putText(frame_copy, 'Hand not found...', (200, 400), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)

    # Draw the ROI on frame copy
    cv2.rectangle(frame_copy, (region_left, region_top), (region_right, region_bottom), (255,128,0), 3)
    
    cv2.putText(frame_copy, "Sign Language Recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    
    frames_count = frames_count + 1

    # Shows the frame with the segmented hand...
    cv2.imshow("Sign Detection", frame_copy)

    # Escape key is the only way to close the application
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        flag = False
#----------------------------------------------------------------------------

# Close the camera & all the other applications
cv2.destroyAllWindows()
cam.release()