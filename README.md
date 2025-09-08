
# Robot Drive Assist

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![Platforms](https://img.shields.io/badge/Platform-Raspberry_Pi-blue.svg)

**Robot Drive Assist** is an embedded project running on a Raspberry Pi-based robot, designed for real-time lane detection and obstacle avoidance. It combines a sliding window technique, a lightweight YOLO11n NCNN model, and PID control algorithms to autonomously navigate a two-lane track while avoiding obstacles.

## Features

- 🚗 Robust lane detection using sliding windows and polynomial fitting
- 🖼️ YOLO-based object detection and obstacle avoidance
- 🎛️ PID control for smooth motor speed adjustments  
- 🔧 Modular Python codebase with configurable parameters  

## Run Locally

### Hardware Setup

The following hardware components are required to run this project:

- Raspberry Pi 4B or newer
- Raspberry Pi Camera Module V2
- 4WD robot car chassis
- L298N Motor driver

### Code Setup

Clone the project

```bash
git clone https://github.com/rejoytm/robot-drive-assist.git
```

Go to the project directory

```bash
cd robot-drive-assist
```

Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

> Note: Some libraries like picamera2 and gpiozero are hardware-specific and will only work on a Raspberry Pi.

Run the main script

```bash
python src/main.py
```


## Configuration

All configurable parameters, such as sliding window settings and frame sizes, are stored in src/config.py. Adjust them as needed for your setup and environment.
