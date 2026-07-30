from machine import Pin, PWM
from time import sleep

SERVO_PINS = [10, 11, 12, 13, 14, 15]

positions = {
    "0":3276,  # Full Speed one Direction
    "90":4915, # Stop
    "180":6553 # Full Speed other direction
}

while True:
    for i in SERVO_PINS:
        try:
            servo = PWM(Pin(i), freq=50)
            
            print(f"Servo {i} Stop")
            servo.duty_u16(4915)   # stop
            sleep(0.8) # This is the duration; 2 seems to be around 360, so 1 around 180
            
            print(f"Servo {i} Full Speed one Direction")
            servo.duty_u16(3276)   
            sleep(0.8)
            
            print(f"Servo {i} Full Speed other Direction")
            servo.duty_u16(6553)
            sleep(0.8)
            
            servo.deinit()
            sleep(2.0)
        
        except Exception as e:
            servo.deinit();
        finally:
            servo.deinit();