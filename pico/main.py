from machine import Pin
import sys
import time
import select

RELAY_PIN = 10
ACTIVE_LOW = True

relay = Pin(RELAY_PIN, Pin.OUT)

def relay_on():
    relay.value(0 if ACTIVE_LOW else 1)

def relay_off():
    relay.value(1 if ACTIVE_LOW else 0)

relay_off()

poll = select.poll()
poll.register(sys.stdin, select.POLLIN)

while True:
    events = poll.poll(50)
    if events:
        line = sys.stdin.readline().strip().upper()

        if line == "RUN":
            relay_on()
            print("OK RUN")
        elif line == "STOP":
            relay_off()
            print("OK STOP")
        elif line == "PING":
            print("PONG")
        else:
            print("ERR")

    time.sleep(0.01)