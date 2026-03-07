## AI Conveyor Belt Sorting System

An intelligent conveyor belt system that uses computer vision, sensors, and embedded control to detect, classify, measure, and sort objects automatically.

This project integrates:
	•	Raspberry Pi (vision processing + system logic)
	•	Raspberry Pi Pico (real-time hardware control)
	•	YOLO object detection
	•	Infrared break-beam sensors
	•	Ultrasonic sensors
	•	Servo-based diverter
	•	Relay-controlled conveyor motor

The system detects objects on a conveyor belt, measures their physical dimensions, analyzes them using AI, and then routes them to the correct destination.

⸻

## System Architecture

Camera
   │
   ▼
Raspberry Pi (AI + Logic)
   │
   │ USB Serial
   ▼
Raspberry Pi Pico (Real-time Control)
   │
   ├── Relay → Conveyor Belt
   ├── Servo → Diverter Gate
   ├── IR Break Beam Sensors
   └── Ultrasonic Sensors

The Raspberry Pi handles AI and decision-making, while the Pico manages real-time hardware control.

⸻

## Hardware Components

Processing
	•	Raspberry Pi
	•	Raspberry Pi Pico

Sensors
	•	2 × IR Break-Beam Sensors (object detection + timing)
	•	2 × Ultrasonic Sensors (dimension measurement)

Actuators
	•	SC90 Micro Servo (sorting gate)
	•	Relay Module (conveyor motor control)

Vision
	•	USB Camera

Conveyor
	•	24V Stainless Steel Conveyor Belt
	•	Speed: ~50 mm/s

⸻

## GPIO Pin Configuration

Raspberry Pi

Physical Pin	GPIO	Function
Pin 2	5V	Power
Pin 6	GND	Ground
Pin 12	GPIO18	Servo signal
Pin 13	GPIO17	Ultrasonic 1 TRIG
Pin 15	GPIO22	Ultrasonic 1 ECHO
Pin 16	GPIO23	Ultrasonic 2 TRIG
Pin 18	GPIO24	Ultrasonic 2 ECHO
Pin 22	GPIO25	IR Sensor 1
Pin 29	GPIO5	IR Sensor 2


⸻

## Object Measurement Method

Object length is calculated using beam interruption timing:

Timer1 = time IR1 beam is broken
Timer2 = time conveyor is stopped

Object Length = (Timer1 - Timer2) × Belt Speed

Example:

Beam broken duration: 8 seconds
Conveyor stopped: 2 seconds
Movement time: 6 seconds

Length = 6 × 50 mm/s = 300 mm


⸻

## Sorting Logic

After analysis, the object is routed via a servo gate.

User-defined rules determine routing.

Example rule file:

Object: apple
Color: red
Width: 50
Length: 80
Thickness: 20
Min_conf: 0.4
Path: C5

The system interprets the path and directs the object accordingly.

⸻

## AI Object Detection

The system uses YOLO (Ultralytics) for object detection.

Each object is analyzed over 600 frames to obtain:
	•	object classification
	•	average confidence
	•	color estimation
	•	detection stability

Model used:

YOLOv11 Nano


⸻

## Conveyor Control

The Raspberry Pi communicates with the Pico via USB serial.

Commands:

RUN
STOP
PING

Example response:

OK RUN
OK STOP
PONG


⸻

## Installation

Clone repository

git clone https://github.com/yourusername/ai-conveyor-sorter.git
cd ai-conveyor-sorter

Install dependencies

pip install ultralytics opencv-python numpy pyserial RPi.GPIO

⸻

License

MIT License

⸻
