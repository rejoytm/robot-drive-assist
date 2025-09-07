import cv2
import numpy as np

from config import (
    DEBUG,
    SLIDING_WINDOW_HALF_WIDTH,
    SLIDING_WINDOW_HEIGHT,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WARPED_FRAME_WIDTH,
    WARPED_FRAME_HEIGHT,
    WARPED_LANE_WIDTH,
    WARPED_VEHICLE_X,
    LANE_LINES
)
from utils import (
    clamp_value,
    map_value,
    color_mask,
    warp_perspective
)

# Masks an image based on config-defined lane line color ranges
def apply_lane_lines_mask(frame):
    height, width, _ = frame.shape
    lane_line_mask = np.zeros((height, width), dtype=np.uint8)

    for lane_line in LANE_LINES:
        mask = color_mask(frame, lane_line["mask_color_range"][0], lane_line["mask_color_range"][1])
        lane_line_mask = cv2.bitwise_or(lane_line_mask, mask)

    return lane_line_mask 

# Finds points along a lane line mask using a sliding window from a given start coordinate
def find_points_using_sliding_window(frame, base_x, base_y=0):
    sliding_window_eval_y = frame.shape[0] - base_y
    points = []
    annotations_frame = frame.copy()

    while sliding_window_eval_y > 0:
        left_limit = base_x - SLIDING_WINDOW_HALF_WIDTH
        right_limit = base_x + SLIDING_WINDOW_HALF_WIDTH
        top_limit = sliding_window_eval_y - SLIDING_WINDOW_HEIGHT
        bottom_limit = sliding_window_eval_y

        window = frame[top_limit:bottom_limit, left_limit:right_limit]     
        sum = np.sum(window, axis=0)

        if (len(sum) == 0):      
            # Shift right if no pixels found in the window
            base_x += SLIDING_WINDOW_HALF_WIDTH
            sliding_window_eval_y -= SLIDING_WINDOW_HEIGHT
            continue

        argmax = np.argmax(sum) + base_x - SLIDING_WINDOW_HALF_WIDTH
        try:
            if (frame[sliding_window_eval_y - 1, argmax] == 0):
                # Skip if the peak column has no active pixels
                sliding_window_eval_y -= SLIDING_WINDOW_HEIGHT 
                continue
        except Exception as _:
            # Skip if out of frame bounds
            sliding_window_eval_y -= SLIDING_WINDOW_HEIGHT
            continue
        
        points.append((argmax, sliding_window_eval_y))

        # Draw window rectangle and peak point for debugging
        cv2.rectangle(annotations_frame, (left_limit, bottom_limit), (right_limit, top_limit), (255, 0, 255), 2)
        cv2.circle(annotations_frame, (argmax, sliding_window_eval_y), 4, (0, 255, 0), -1)
        
        base_x = argmax
        sliding_window_eval_y -= SLIDING_WINDOW_HEIGHT

    if DEBUG:
        cv2.imshow("Sliding window annotations", annotations_frame)
    return points

# Plots lane lines by fitting a polynomial to center line points and offsetting them
def plot_lane_lines(frame, c_line_points):
    if (len(c_line_points) == 0):
        return frame
    
    base_frame = frame
    c_line_fit = np.polyfit([point[1] for point in c_line_points], [point[0] for point in c_line_points], 2)
    y_vals = np.arange(WARPED_FRAME_HEIGHT)

    c_x_vals = np.polyval(c_line_fit, y_vals)
    ref_points = np.array(list(zip(c_x_vals, y_vals)), dtype=np.int32)
    for point in ref_points:
        cv2.circle(base_frame, point, 4, (255, 255, 255), -1)

    l_x_vals = [x - WARPED_LANE_WIDTH for x in c_x_vals]
    l_points = np.array(list(zip(l_x_vals, y_vals)), dtype=np.int32)
    for point in l_points:
        cv2.circle(base_frame, point, 4, (255, 255, 255), -1)

    r_x_vals = [x + WARPED_LANE_WIDTH for x in c_x_vals]
    r_points = np.array(list(zip(r_x_vals, y_vals)), dtype=np.int32)
    for point in r_points:
        cv2.circle(base_frame, point, 4, (255, 255, 255), -1)

    return base_frame

# Detects lane lines in an image and returns the fitted center line and lane offset
def detect_lanes(image, result_queue):
    """
    We assume a two-lane setup in the following manner: 
    - A solid left line (l), a dashed center line (c), and a solid right line (r). 
    - The vehicle drives in the right-hand lane (i.e. between the center and right line). 
    - In the bird's-eye view, knowing one line's points lets us estimate the other lines by applying an offset (WARPED_LANE_WIDTH).
    """    

    frame = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
    warped_frame = warp_perspective(frame)

    # Extract lane lines mask from the warped frame and erode it to reduce noise
    lane_lines_mask = apply_lane_lines_mask(warped_frame)
    lane_lines_mask = cv2.erode(lane_lines_mask,  np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(lane_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists for storing solid and dashed line contours
    solid_line_contours = []
    dashed_line_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if (area < 200):
            # Small contours are likely noise
            continue
        elif (area < 900):
            # Medium contours are likely dashed lines (center line)
            dashed_line_contours.append(contour)
        else:
            # Large contours are likely solid lines (left or right lines)
            solid_line_contours.append(contour)
    
    # Initialize lists for left and right line contours
    l_line_contours = []
    r_line_contours = []

    # Categorize solid line contours as belonging to the left or right line based on their position
    for contour in solid_line_contours:
        bottom_point = max(contour, key=lambda x: x[0][1])
        bottom_x = bottom_point[0][0]
        bottom_y = bottom_point[0][1]

        if bottom_y < WARPED_FRAME_HEIGHT // 2:
            # Ignore contours in the upper half of the frame as they cannot be reliably categorized
            continue
        else:
            # Classify contours in the lower half of the frame based on their horizontal position
            if bottom_x < WARPED_FRAME_WIDTH // 2: # Left half of the frame
                l_line_contours.append(contour)
            else: # Right half of the frame
                r_line_contours.append(contour)   

    # Create masks for left, center, and right lane lines
    l_line_mask = np.zeros_like(lane_lines_mask)
    c_line_mask = np.zeros_like(lane_lines_mask)
    r_line_mask = np.zeros_like(lane_lines_mask)
    cv2.drawContours(l_line_mask, l_line_contours, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(c_line_mask, dashed_line_contours, -1, (255, 255, 255), cv2.FILLED)
    cv2.drawContours(r_line_mask, r_line_contours, -1, (255, 255, 255), cv2.FILLED)    

    """
    We estimate center line points by applying a sliding window on visible line contours. 
    Solid line contours are more reliable than dashed lines for the sliding window approach, 
    so we check for them in the following order:
    1. Solid right line (since the vehicle is driving on the right lane).
    2. Solid left line.
    3. Dashed center line.
    """    

    # Initialize list for center line points
    c_line_points = []

    if (len(r_line_contours)): # Right line is visible
        r_line_points = find_points_using_sliding_window(r_line_mask, LANE_LINES[2]["initial_x"])
        c_line_points = [[x - WARPED_LANE_WIDTH, y] for x, y in r_line_points]
    elif (len(l_line_contours)): # Left line is visible
        l_line_points = find_points_using_sliding_window(l_line_mask, LANE_LINES[0]["initial_x"], 100)
        c_line_points = [[x + WARPED_LANE_WIDTH, y] for x, y in l_line_points]
    elif (len(dashed_line_contours)): # Only center line is visible
        c_line_points = []
        for contour in dashed_line_contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                c_line_points.append((cx, cy))

    # Debugging: Display the warped image, lane lines mask, and the plot of predicted lane lines
    if (DEBUG):
        cv2.imshow("Warped Image", warped_frame)    
        cv2.imshow("Lane Lines Mask", lane_lines_mask)    
        predicted_lines_plot = np.zeros_like(lane_lines_mask)
        predicted_lines_plot = plot_lane_lines(predicted_lines_plot, c_line_points)
        cv2.imshow("Predicted Lines Plot", predicted_lines_plot)    
        cv2.waitKey(0)

    # If center line points are found, fit a curve to them and calculate the lane offset
    if (len(c_line_points)):
        # Fit a polynomial to the center line points
        c_line_fit = np.polyfit([point[1] for point in c_line_points], [point[0] for point in c_line_points], 2)
        
        # Determine the center of the lane at the bottom of the warped frame
        decision_eval_y = WARPED_FRAME_HEIGHT
        lane_half_width = WARPED_LANE_WIDTH // 2
        lane_center = int(np.polyval(c_line_fit, decision_eval_y)) + lane_half_width  # Add half a lane width (vehicle is in the right-hand lane)

        # Calculate the lane offset (vehicle's offset from the lane center)
        lane_offset = clamp_value(WARPED_VEHICLE_X - lane_center, -lane_half_width, lane_half_width)
        lane_offset = round(map_value(lane_offset, -lane_half_width, lane_half_width, -63, 63))  # Map lane offset for PID control

        result_queue.put((c_line_fit, lane_offset))        
        return c_line_fit, lane_offset
    else: # No center line points found
        result_queue.put((None, None))        
        return None, None