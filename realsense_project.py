import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2


class RealsenseCamera:
    def __init__(self):

        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))
        # print("\ndevice_product_line = {}\n".format(device_product_line))



        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        config_start = self.pipeline.start(config)
        self.depth_scale = config_start.get_device().first_depth_sensor().get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        
        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        self.depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        
        # Convert images to numpy arrays
        # distance = depth_frame.get_distance(int(50),int(50))
        # print("distance", distance)
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # cv2.imshow("Colormap", depth_colormap)
        # cv2.imshow("depth img", depth_image)

        return True, color_image, depth_image
    
    def release(self):
        self.pipeline.stop()
        
        #print(depth_image)
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        # Stack both images horizontally
        
        #images = np.hstack((color_image, depth_colormap)


dc = RealsenseCamera()

depth_list = []

## counter = 0
while True:
# while (counter != 5):
    ## counter = counter + 1
    #Get frame in realtime from realsense camera
    ret, color_frame, depth_frame = dc.get_frame_stream()

    color = dc.depth_colormap
    # color = np.asanyarray(color_frame.get_data())
    
    # Standard OpenCV boilerplate for running the net:
    height, width = color.shape[:2]
    expected = 300
    aspect = width / height
    resized_image = cv2.resize(color, (round(expected * aspect), expected))
    crop_start = round(expected * (aspect - 1) / 2)
    crop_img = resized_image[0:expected, crop_start:crop_start+expected]

    prototxt_path = r"G:\coding_folder\Python\Python3\jupyter\MobileNetSSD_deploy.prototxt.txt"
    model_path = r"G:\coding_folder\Python\Python3\jupyter\MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    inScaleFactor = 0.007843
    meanVal       = 127.53
    classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair",
                "cow", "diningtable", "dog", "horse",
                "motorbike", "person", "pottedplant",
                "sheep", "sofa", "train", "tvmonitor")

    blob = cv2.dnn.blobFromImage(crop_img, inScaleFactor, (expected, expected), meanVal, False)
    net.setInput(blob, "data")
    detections = net.forward("detection_out")

    label = detections[0,0,0,1]
    conf  = detections[0,0,0,2]
    xmin  = detections[0,0,0,3]
    ymin  = detections[0,0,0,4]
    xmax  = detections[0,0,0,5]
    ymax  = detections[0,0,0,6]

    className = classNames[int(label)]
    cv2.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)), (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)

    cv2.putText(crop_img, className, (int(xmin * expected), int(ymin * expected) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255))

    #Detected color image
    cv2.imshow("Color Frame", crop_img)

    scale = height / expected
    xmin_depth = int((xmin * expected + crop_start) * scale)
    ymin_depth = int((ymin * expected) * scale)
    xmax_depth = int((xmax * expected + crop_start) * scale)
    ymax_depth = int((ymax * expected) * scale)
    xmin_depth,ymin_depth,xmax_depth,ymax_depth

    cv2.rectangle(depth_frame, (xmin_depth, ymin_depth), (xmax_depth, ymax_depth), (255, 255, 255), 2)
    
    #Showing depth frame
    cv2.imshow("Depth Frame", depth_frame)

    
    depth = np.asanyarray(depth_frame)
    # Crop depth data:
    depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)

    # Get data scale from the device and convert to meters
    depth_scale = dc.depth_scale
    depth = depth * depth_scale
    dist,_,_,_ = cv2.mean(depth)
    depth_list.append(dist)
    print("Detected a {0} {1:.3} meters away.".format(className, dist))
    print(dist)

        
    key = cv2.waitKey(1) & 0xFF
    ## key = cv2.waitKey(2000) & 0xFF
    ##    print("\nKey Pressed = {}\n".format(key))
    if ((key == ord('q')) or (key == 27)):
        ## print("\n\"q\" or esc has been pressed.\n")
        cv2.destroyAllWindows()
        break


frames_list = list(range(0, len(depth_list)))
## print("\ndepth_list = {}\n".format(depth_list))
plt.plot(frames_list,depth_list)
plt.title('Depth vs frame number')
plt.show()
