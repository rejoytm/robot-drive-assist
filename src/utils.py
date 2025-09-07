import cv2

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WARPED_FRAME_WIDTH,
    WARPED_FRAME_HEIGHT,
    ORIGINAL_PERSPECTIVE_POINTS,
    WARPED_PERSPECTIVE_POINTS,
    WARP_PERSPECTIVE_BG_COLOR
)

# Limits a value to stay within a range
def clamp_value(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Maps a value from one range to another
def map_value(x, a, b, c, d):
    return (x - a) * (d - c) / (b - a) + c

# Masks an image based on an HSV color range
def color_mask(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

# Warps an image and fills empty spaces with a color based on config-defined parameters
def warp_perspective(frame, bg_color=WARP_PERSPECTIVE_BG_COLOR):
    matrix = cv2.getPerspectiveTransform(ORIGINAL_PERSPECTIVE_POINTS, WARPED_PERSPECTIVE_POINTS) 
    return cv2.warpPerspective(frame, matrix, (WARPED_FRAME_WIDTH, WARPED_FRAME_HEIGHT), borderValue=bg_color)

# Reverts a warped image to its original perspective based on config-defined parameters
def unwarp_perspective(frame):
    matrix = cv2.getPerspectiveTransform(WARPED_PERSPECTIVE_POINTS, ORIGINAL_PERSPECTIVE_POINTS) 
    return cv2.warpPerspective(frame, matrix, (FRAME_WIDTH, FRAME_HEIGHT))