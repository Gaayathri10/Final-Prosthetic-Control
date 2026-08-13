from machine import Pin, PWM
from time import sleep
import serial as ser


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

    THUMB_SERVO_PIN = 15
    INDEX_SERVO_PIN = 14
    MIDDLE_FINGER_SERVO_PIN = 13
    RING_FINGER_SERVO_PIN = 12
    PINKY_SERVO_PIN = 11
    WRIST_SERVO_PIN = 10

    def __init__(self):

        self.thumb = ContinuousServo(
            pin=self.THUMB_SERVO_PIN,
            stop=4915,
            speed=220,
            seconds_per_degree=0.0068
        )

        self.index_finger = ContinuousServo(
            pin=self.INDEX_SERVO_PIN,
            stop=4910,
            speed=220,
            seconds_per_degree=0.0066
        )

        self.middle_finger = ContinuousServo(
            pin=self.MIDDLE_FINGER_SERVO_PIN,
            stop=4920,
            speed=220,
            seconds_per_degree=0.0069
        )

        self.ring_finger = ContinuousServo(
            pin=self.RING_FINGER_SERVO_PIN,
            stop=4915,
            speed=220,
            seconds_per_degree=0.0068
        )

        self.pinky_finger = ContinuousServo(
            pin=self.PINKY_SERVO_PIN,
            stop=4910,
            speed=220,
            seconds_per_degree=0.0066
        )

        self.wrist = ContinuousServo(
            pin=self.WRIST_SERVO_PIN,
            stop=4915,
            speed=220,
            seconds_per_degree=0.0068
        )

        self.last_gesture = None

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

    def move_smooth(self, thumb, duty):

        self.thumb.move_to(thumb)
        self.index_finger.move_to(duty)
        self.middle_finger.move_to(duty)
        self.ring_finger.move_to(duty)
    # ------------------------------------------------------

    def neutral(self):

        self.move_smooth(
            self.MIN,
            self.MIN
        )

    # ------------------------------------------------------

    def pinch(self):

        self.thumb.move_to(self.MAX_THUMB)

        sleep(0.1)

        self.index_finger.move_to(self.MAX)
        self.middle_finger.move_to(20)


    # ------------------------------------------------------

    def grasp(self):

        self.move_smooth(
            self.MAX_THUMB,
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
            self.MAX
        )

        sleep(0.5)

        self.neutral()

    # ------------------------------------------------------

    def shutdown(self):

        self.thumb.stop()
        self.index_finger.stop()
        self.middle_finger.stop()
        self.ring_finger.stop()
        self.pinky_finger.stop()
        self.wrist.stop()

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