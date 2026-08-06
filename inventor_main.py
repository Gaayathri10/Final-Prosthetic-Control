from machine import Pin, PWM
from time import sleep
import serial


# ==========================================================
# Continuous Rotation Servo
# ==========================================================

class ContinuousServo:
    def __init__(self, pin,
                 stop=4915,
                 speed=220,
                 seconds_per_degree=0.0065):

        self.pwm = PWM(Pin(pin))
        self.pwm.freq(50)

        self.stop_pwm = stop
        self.speed = speed
        self.seconds_per_degree = seconds_per_degree

        # Estimated position
        self.position = 0

        self.stop()

    def stop(self):
        self.pwm.duty_u16(self.stop_pwm)

    def clockwise(self):
        self.pwm.duty_u16(self.stop_pwm - self.speed)

    def counter_clockwise(self):
        self.pwm.duty_u16(self.stop_pwm + self.speed)

    def move_to(self, target):

        target = max(0, min(180, target))

        if target == self.position:
            return

        delta = target - self.position

        if delta > 0:
            self.clockwise()
        else:
            self.counter_clockwise()

        sleep(abs(delta) * self.seconds_per_degree)

        self.stop()

        self.position = target

    def home(self):
        self.move_to(0)

class RoboticHand:

    MIN = 0
    MAX = 180
    MAX_THUMB = 90

    def __init__(self):

        self.thumb = ContinuousServo(
            pin=8,
            stop=4915,
            speed=220,
            seconds_per_degree=0.0068
        )

        self.servo2 = ContinuousServo(
            pin=10,
            stop=4910,
            speed=220,
            seconds_per_degree=0.0066
        )

        self.servo3 = ContinuousServo(
            pin=12,
            stop=4920,
            speed=220,
            seconds_per_degree=0.0069
        )

        self.last_gesture = None

    # ------------------------------------------------------

    def update(self, gesture):

        if gesture == self.last_gesture:
            return

        # Always return to neutral first
        self.neutral()
        sleep(0.1)

        if gesture == 0:
            self.neutral()

        elif gesture == 1:
            self.pinch()

        elif gesture == 2:
            self.grasp()

        elif gesture == 3:
            self.pinch()

        elif gesture == 4:
            self.clench_unclench()

        elif gesture == 5:
            self.move_thumb()

        self.last_gesture = gesture

    # ------------------------------------------------------

    def move_smooth(self, thumb, s2, s3):

        self.thumb.move_to(thumb)
        self.servo2.move_to(s2)
        self.servo3.move_to(s3)

    # ------------------------------------------------------

    def neutral(self):

        self.move_smooth(
            self.MIN,
            self.MIN,
            self.MIN
        )

    # ------------------------------------------------------

    def pinch(self):

        self.thumb.move_to(self.MAX_THUMB)

        sleep(0.1)

        self.servo2.move_to(self.MAX)

        self.servo3.move_to(20)

    # ------------------------------------------------------

    def grasp(self):

        self.move_smooth(
            self.MAX_THUMB,
            self.MAX,
            self.MAX
        )

    # ------------------------------------------------------

    def move_thumb(self):

        self.thumb.move_to(self.MAX_THUMB)

        sleep(0.5)

        self.thumb.move_to(self.MIN)

    # ------------------------------------------------------

    def clench_unclench(self):

        self.move_smooth(
            self.MAX_THUMB,
            self.MAX,
            self.MAX
        )

        sleep(0.5)

        self.neutral()

    # ------------------------------------------------------

    def shutdown(self):

        self.thumb.stop()
        self.servo2.stop()
        self.servo3.stop()

PORT = "/dev/tty50"
BAUDRATE = 9600

if __name__ == "__main__":
    hand = RoboticHand()

    while True:
        data = ser.Serial(PORT, BAUDRATE, timeout=100)
        if not data:
            continue
            
        code = data[0]
        gesture = gesture_map.get(code, f"Unknown({code})")
        print("Received gesture code:", code, "=>", gesture)