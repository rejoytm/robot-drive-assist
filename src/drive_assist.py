import multiprocessing

from config import DESIRED_LANE_OFFSET, DESIRED_MIO_DISTANCE
from lane_detection import detect_lanes
from object_detection import detect_objects, find_mio

# Processes an image and returns lane offset and distance to the closest in-lane object (MIO)
def get_lane_offset_and_mio_distance(image):
    lane_detection_queue = multiprocessing.Queue()
    object_detection_queue = multiprocessing.Queue()

    # Run lane and object detection in parallel
    lane_detection_process = multiprocessing.Process(target=detect_lanes, args=(image, lane_detection_queue))
    object_detection_process = multiprocessing.Process(target=detect_objects, args=(image, object_detection_queue))
    lane_detection_process.start()
    object_detection_process.start()

    # Wait for both processes to complete and retrieve results
    lane_detection_process.join()
    object_detection_process.join()
    lane_detection_result = lane_detection_queue.get()
    object_detection_result = object_detection_queue.get()

    # Get lane center line and offset
    c_line_fit, lane_offset = lane_detection_result

    if c_line_fit is None:
        # No lanes detected
        return None, None

    # Get distance to closest in-lane object (MIO)
    _, mio_distance = find_mio(c_line_fit, object_detection_result)

    return lane_offset, mio_distance