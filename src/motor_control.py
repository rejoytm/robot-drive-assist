from gpiozero import Motor

from config import MIN_MOTOR_SPEED, MAX_MOTOR_SPEED
from utils import clamp_value, map_value

def initialize_motors():
    motor_l = Motor(17, 18, pwm=True)
    motor_r = Motor(22, 23, pwm=True)
    return motor_l, motor_r

def set_motor_speeds(motor_l, motor_l_speed, motor_r, motor_r_speed):
    # Limit motor speeds to config-defined range
    motor_l_speed = clamp_value(motor_l_speed, MIN_MOTOR_SPEED, MAX_MOTOR_SPEED)
    motor_r_speed = clamp_value(motor_r_speed, MIN_MOTOR_SPEED, MAX_MOTOR_SPEED)

    # Scale motor speeds from 0-255 to 0-1 for PWM
    motor_l_pwm = map_value(motor_l_speed, 0, 255, 0, 1)
    motor_r_pwm = map_value(motor_r_speed, 0, 255, 0, 1)

    motor_l.forward(motor_l_pwm)
    motor_r.forward(motor_r_pwm)

    return motor_l_pwm, motor_r_pwm
