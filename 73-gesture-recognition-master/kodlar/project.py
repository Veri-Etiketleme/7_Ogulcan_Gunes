import cv2
import numpy as np
import math
from message import *
from counter import *


# Variables & parameters
ans_list=[]
hsv_thresh_lower=150
gaussian_ksize=11
gaussian_sigma=0
morph_elem_size=13
median_ksize=3
capture_box_count=9 #number of small boxes
capture_box_dim=20  #small box size
capture_box_sep_x=8  #x coordinate distance between small boxes
capture_box_sep_y=18  #y coordinate distance between small boxes
capture_pos_x=500  #start of small boxes
capture_pos_y=150  #start of small boxes y coordinate
cap_region_x_begin=0.5 # start point/total width
cap_region_y_end=0.8# start point/total width
finger_thresh_l=2.0
finger_thresh_u=3.8
radius_thresh=0.04 # factor of width of full frame
first_iteration=True
finger_ct_history=[0,0]

# ------------------------ Function declarations ------------------------ #

# 1. Hand capture histogram
def hand_capture(frame_in,box_x,box_y):
    hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    ROI = np.zeros([capture_box_dim*capture_box_count,capture_box_dim,3], dtype=hsv.dtype)  #roi[y1:y2,x1:x2]
    for i in xrange(capture_box_count):
        ROI[i*capture_box_dim:i*capture_box_dim+capture_box_dim,0:capture_box_dim] = hsv[box_y[i]:box_y[i]+capture_box_dim,box_x[i]:box_x[i]+capture_box_dim]
    hand_hist = cv2.calcHist([ROI],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist,hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

# 2. Filters and threshold
def hand_threshold(frame_in,hand_hist):
    frame_in=cv2.medianBlur(frame_in,3)
    hsv=cv2.cvtColor(frame_in,cv2.COLOR_BGR2HSV)
    hsv[0:int(cap_region_y_end*hsv.shape[0]),0:int(cap_region_x_begin*hsv.shape[1])]=0 # Right half screen only
    hsv[int(cap_region_y_end*hsv.shape[0]):hsv.shape[0],0:hsv.shape[1]]=0
    back_projection = cv2.calcBackProject([hsv], [0,1],hand_hist, [00,180,0,256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_elem_size,morph_elem_size))
    cv2.filter2D(back_projection, -1, disc, back_projection)
    back_projection=cv2.GaussianBlur(back_projection,(gaussian_ksize,gaussian_ksize), gaussian_sigma)
    back_projection=cv2.medianBlur(back_projection,median_ksize)
    ret, thresh = cv2.threshold(back_projection, hsv_thresh_lower, 255, 0)
    
    return thresh

# 3. Find hand contour
def hand_contour_find(contours):
    max_area=0
    largest_contour=-1
    for i in range(len(contours)):
        cont=contours[i]
        area=cv2.contourArea(cont)
        if(area>max_area):
            max_area=area
            largest_contour=i
    if(largest_contour==-1):
        return False,0
    else:
        h_contour=contours[largest_contour]
        return True,h_contour


#fingers_detction
def mark_fingers(hull,pt,cnt):
    if(1):
        defects = cv2.convexityDefects(cnt,hull)
        mind=0
        maxd=0
        count_defects = 0
        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start=tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt,pt,True)
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1
        
                
            #cv2.line(img,start,end,[0,255,0],2)
            #cv2.circle(img,far,5,[0,0,255],-1)
        if len(ans_list)<35:
            ans_list.append(count_defects)
        else:
            ans_fing=max_freq(ans_list)
            ans_list[:] = []
            message(ans_fing+1)
            print (ans_fing+1)
         
        i=0

    return i
        
        
            
    
    

# 5. Mark hand center circle

def mark_hand_center(frame_in,cont):    
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(cont)
    for ind_y in xrange(int(y+0.3*h),int(y+0.8*h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results)
        for ind_x in xrange(int(x+0.3*w),int(x+0.6*w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results)
            dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    if(max_d>radius_thresh*frame_in.shape[1]):
        thresh_score=True
        cv2.circle(frame_in,pt,int(max_d),(255,0,0),2)
    else:
        thresh_score=False
    return frame_in,pt,max_d,thresh_score

# 6. Find and display gesture

# 7. Remove bg from image

def remove_bg(frame):
    fg_mask=bg_model.apply(frame)
    kernel = np.ones((3,3),np.uint8)
    fg_mask=cv2.erode(fg_mask,kernel,iterations = 1)
    frame=cv2.bitwise_and(frame,frame,mask=fg_mask)
    return frame

# ------------------------ BEGIN ------------------------ #

# Camera
camera = cv2.VideoCapture(0)
capture_done=0
bg_captured=0


while(1):
    # Capture frame from camera
    ret, frame = camera.read()
    frame=cv2.bilateralFilter(frame,5,50,100)
    # Operations on the frame
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame,(int(cap_region_x_begin*frame.shape[1]),0),(frame.shape[1],int(cap_region_y_end*frame.shape[0])),(255,0,0),1)
    frame_original=np.copy(frame)
    if(bg_captured):
        fg_frame=remove_bg(frame)
        
    if (not (capture_done and bg_captured)):
        if(not bg_captured):
            cv2.putText(frame,"Remove hand from the frame and press 'b' to capture background",(int(0.05*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        else:
            cv2.putText(frame,"Place hand inside boxes and press 'c' to capture hand histogram",(int(0.08*frame.shape[1]),int(0.97*frame.shape[0])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1,8)
        first_iteration=True
        finger_ct_history=[0,0]
        #box_pos_x,box_pos_y used for coordinates of small boxes(i.e 9 boxes)
        box_pos_x=np.array([capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x,capture_pos_x,capture_pos_x+capture_box_dim+capture_box_sep_x,capture_pos_x+2*capture_box_dim+2*capture_box_sep_x],dtype=int)
        box_pos_y=np.array([capture_pos_y,capture_pos_y,capture_pos_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+capture_box_dim+capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y,capture_pos_y+2*capture_box_dim+2*capture_box_sep_y],dtype=int)
        for i in range(capture_box_count):
            cv2.rectangle(frame,(box_pos_x[i],box_pos_y[i]),(box_pos_x[i]+capture_box_dim,box_pos_y[i]+capture_box_dim),(255,0,0),1)
    else:
        frame=hand_threshold(fg_frame,hand_histogram)
        contour_frame=np.copy(frame)
        contours,hierarchy=cv2.findContours(contour_frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        found,hand_contour=hand_contour_find(contours)
        hand_contour = cv2.approxPolyDP(hand_contour,0.01*cv2.arcLength(hand_contour,True),True)
        if(found):
            hand_convex_hull=cv2.convexHull(hand_contour,returnPoints = False)
            frame,hand_center,hand_radius,hand_size_score=mark_hand_center(frame_original,hand_contour)
            if(hand_size_score):
                ti=mark_fingers(hand_convex_hull,hand_center,hand_contour)
                
        else:
            frame=frame_original

    # Display frame in a window
    cv2.imshow('Hand Gesture Recognition v1.0',frame)
    interrupt=cv2.waitKey(10)
    
    # Quit by pressing 'q'
    if  interrupt & 0xFF == ord('q'):
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        if(bg_captured):
            capture_done=1
            hand_histogram=hand_capture(frame_original,box_pos_x,box_pos_y)
    # Capture background by pressing 'b'
    elif interrupt & 0xFF == ord('b'):
        bg_model = cv2.BackgroundSubtractorMOG2(0,10)
        bg_captured=1
    # Reset captured hand by pressing 'r'
    elif interrupt & 0xFF == ord('r'):
        capture_done=0
        bg_captured=0
        
# Release camera & end program
camera.release()
cv2.destroyAllWindows()
