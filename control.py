import serial
import time

def connect_to_serial_port(port='COM5', baudrate=115200, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        if ser.is_open:
            print(f"Connected to {port} successfully.")
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to {port}: {e}")
        return None

ser = connect_to_serial_port()
if ser:
    # Perform serial communication here

    for i in range(1):
        ser.write(b'100\n') # Simplified byte encoding
        time.sleep(0.5)
    print("reverse")
    for i in range(1):
        ser.write(b'-100\n') # Simplified byte encoding
        time.sleep(0.3)

    ser.close()