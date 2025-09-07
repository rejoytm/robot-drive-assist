import numpy as np

# Debugging mode
DEBUG = False

# Sliding window parameters
SLIDING_WINDOW_HALF_WIDTH = 60
SLIDING_WINDOW_HEIGHT = 50

# Frame sizes
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WARPED_FRAME_WIDTH = 480
WARPED_FRAME_HEIGHT = 720

# Perspective transformation points
ORIGINAL_PERSPECTIVE_POINTS = np.float32([
    (110, 250), # top-left
    (-180, 400), # bottom-left
    (415, 247), # top-right
    (580, 400) # bottom-right
])

WARPED_PERSPECTIVE_POINTS = np.float32([
    (0, 0), # top-left
    (0, WARPED_FRAME_HEIGHT), # bottom-left
    (WARPED_FRAME_WIDTH, 0), # top-right
    (WARPED_FRAME_WIDTH, WARPED_FRAME_HEIGHT) # bottom-right
])

# Background color to fill empty areas in warped perspective
WARP_PERSPECTIVE_BG_COLOR = (255, 255, 255)

# Lane width and vehicle x-position in warped perspective
WARPED_LANE_WIDTH = 420 - 210
WARPED_VEHICLE_X = (420 + 210) // 2

# Lane line parameters
LANE_LINES = [
    {
        'name': 'Left', # Label used for debugging
        'mask_color_range': (np.array([0, 0, 200]), np.array([180, 50, 255])), # HSV color range used for masking line
        'initial_x': 20 # X-coordinate where the sliding window starts
    },    
    {
        'name': 'Center',
        'mask_color_range': (np.array([0, 0, 200]), np.array([180, 50, 255])),
        'initial_x': 240
    },
    {
        'name': 'Right',
        'mask_color_range': (np.array([0, 0, 200]), np.array([180, 50, 255])),
        'initial_x': 460
    }
]

# Motor control parameters
MIN_MOTOR_SPEED = 0
MAX_MOTOR_SPEED = 255
BASE_MOTOR_SPEED = 100

# PID control parameters
DESIRED_LANE_OFFSET = 0
DESIRED_MIO_DISTANCE = WARPED_FRAME_HEIGHT