import cv2
from realsense_camera import *
from mask_rcnn import *

# Load Realsense camera and Mask R-CNN
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
    #Get frame in realtime from realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    
    #Get Object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)
    
    #Draw Object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    
    #Show depth info of the Object
    mrcnn.draw_object_info(bgr_frame, depth_frame)
    
    cv2.imshow("Depth frame", depth_frame)
    cv2.imshow("BGR frame", bgr_frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
