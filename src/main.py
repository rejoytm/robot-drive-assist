import time

from config import (
    DEBUG,
    BASE_MOTOR_SPEED,
    DESIRED_LANE_OFFSET,
    DESIRED_MIO_DISTANCE
)
from camera_control import initialize_camera, capture_image
from motor_control import initialize_motors, set_motor_speeds
from pid import PIDController 
from drive_assist import get_lane_offset_and_mio_distance

def main():
    # Set up camera, motors, and PID controllers
    camera = initialize_camera()
    motor_l, motor_r = initialize_motors()
    pid_lane = PIDController(kp=3.5, ki=0.0005, kd=1.2)
    pid_mio = PIDController(kp=4.0, ki=0.0002, kd=1.5)

    while True:
        image = capture_image(camera)
        lane_offset, mio_distance = get_lane_offset_and_mio_distance(image)

        if lane_offset is None:
            # No lanes detected
            continue

        # Adjust motor speeds based on lane offset and distance to MIO
        pid_lane_control = pid_lane.compute(DESIRED_LANE_OFFSET, lane_offset)
        pid_mio_control = pid_mio.compute(DESIRED_MIO_DISTANCE, mio_distance)
        motor_l_speed = BASE_MOTOR_SPEED - pid_lane_control - pid_mio_control
        motor_r_speed = BASE_MOTOR_SPEED + pid_lane_control - pid_mio_control

        motor_l_pwm, motor_r_pwm = set_motor_speeds(motor_l, motor_l_speed, motor_r, motor_r_speed)

        if DEBUG:
            print(f"Motor PWMs - Left: {motor_l_pwm:.2f}, Right: {motor_r_pwm:.2f}")

        time.sleep(0.1)

if __name__ == "__main__":
    main()