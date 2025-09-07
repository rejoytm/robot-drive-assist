from picamera2 import Picamera2

def initialize_camera():
    picam2 = Picamera2()
    picam2_config = picam2.create_video_configuration(
        raw={"size": (1640, 1232)}, main={"size": (640, 480), "format": "XRGB8888"}
    )
    picam2.align_configuration(picam2_config)
    picam2.configure(picam2_config)
    picam2.start()
    return picam2

def capture_image(picam2):
    return picam2.capture_array()
