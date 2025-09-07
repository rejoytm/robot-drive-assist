import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WARPED_FRAME_HEIGHT,
    WARPED_LANE_WIDTH
)
from utils import warp_perspective

object_detection_model = YOLO("models/yolo11n_ncnn_model", task="detect")

# Detects objects in an image and puts bounding boxes in a multiprocessing queue
def detect_objects(image, result_queue):
    image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
    detected_boxes = []
    
    try:
        results = object_detection_model(image)

        for bbox in results.boxes.xyxy:
            xmin, ymin, xmax, ymax = bbox.tolist()
            detected_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)]) 
    except Exception as e:
        print(f"Model Inference Error: {str(e)}")
    finally:
        result_queue.put(detected_boxes)
        return detected_boxes
    
# Computes the distance to a detected in-lane object using its bounding box and the lane center line fit
def compute_distance_to_object(c_line_fit, xmin, xmax, ymax):
    """
    The center line fit (c_line_fit) is defined in the bird's-eye view.
    To compute the distance from the vehicle to the object, 
    we need to determine the object's position in this warped perspective.
    """    

    # Draw the bottom edge of the object on a grayscale frame
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8) 
    cv2.line(frame, (xmin, ymax), (xmax, ymax), 255, 3) 

    # Warp to bird's-eye view and extract contours 
    warped_frame = warp_perspective(frame, bg_color=0) 
    contours, _ = cv2.findContours(warped_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if (len(contours) == 0):
        # No contours found, assume maximum distance
        return WARPED_FRAME_HEIGHT
     
    # Get bounding box of the warped object contour
    warped_xmin, warped_ymin, width, height = cv2.boundingRect(contours[0])
    warped_xmax = warped_xmin + width
    warped_ymax = warped_ymin + height

    # Determine the lane boundaries at the bottom edge of the warped object
    decision_eval_y = warped_ymax
    line_margin = 20 # Shrinks lane boundaries for a stricter in-lane check
    c_line_x = int(np.polyval(c_line_fit, decision_eval_y)) + line_margin # Center lane line position
    r_line_x = c_line_x + WARPED_LANE_WIDTH - line_margin * 2 # Right lane line position

    is_object_in_lane = c_line_x <= warped_xmin <= r_line_x or c_line_x <= warped_xmax <= r_line_x # Assuming vehicle is in the right-hand lane
        
    if (is_object_in_lane):
        distance = WARPED_FRAME_HEIGHT - warped_ymax
        return distance
    else:
        # Object not in lane, assume maximum distance
        return WARPED_FRAME_HEIGHT

# Finds the closest in-lane object (MIO) and its distance
def find_mio(c_line_fit, object_detection_result):
    mio = None
    mio_distance = WARPED_FRAME_HEIGHT

    for detection in object_detection_result:
        xmin, _, xmax, ymax = detection
        distance = compute_distance_to_object(c_line_fit, xmin, xmax, ymax)
        if distance < mio_distance:
            mio_distance = distance
            mio = detection    

    return mio, mio_distance
