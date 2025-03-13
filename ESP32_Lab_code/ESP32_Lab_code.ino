#include <Wire.h>
#include <ESP32Encoder.h>
// MMA8452Q I2C address is 0x1C(28)
#define Addr 0x1C

#define CLK 18 // CLK ENCODER
#define DT 19 // DT ENCODER

int ENA = 15;
int IN1 = 2;
int IN2 = 4;
char incomingByte;
int movement = 0;
uint8_t pwmValue = 0;
long currentPosition = 0; 
ESP32Encoder encoder;
boolean data_stream = false;
int sign = 1;

enum Direction {
  LEFT,
  RIGHT,
  NONE
}; 

void movemotor(int p1,int p2,int num){
    digitalWrite(IN1, p1);
    digitalWrite(IN2, p2); 
    delay(num);
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
  }

void movemotorPWM(uint8_t value, Direction direction) {
  pwmValue = value;
  if (value == 0) {
    digitalWrite(IN1, 0);
    digitalWrite(IN2, 0);
    return; 
  }

  if (direction == LEFT) {
    analogWrite(IN1, value);
    digitalWrite(IN2, 0);
  } else if (direction == RIGHT) {
    digitalWrite(IN1, 0);
    analogWrite(IN2, value);
  } else {
    digitalWrite(IN1, 0);
    digitalWrite(IN2, 0);
  }
 
  }

void setup()
{
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  digitalWrite(ENA, HIGH); 
  // Initialise I2C communication as MASTER
  Wire.begin();
  // Initialise Serial Communication, set baud rate = 9600
  Serial.begin(115200);
  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x2A);
  // StandBy mode
  Wire.write((byte)0x00);
  // Stop I2C Transmission
  Wire.endTransmission();
 
  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x2A);
  // Active mode
  Wire.write(0x01);
  // Stop I2C Transmission
  Wire.endTransmission();
 
  // Start I2C Transmission
  Wire.beginTransmission(Addr);
  // Select control register
  Wire.write(0x0E);
  // Set range to +/- 2g
  Wire.write((byte)0x00);
  // Stop I2C Transmission
  Wire.endTransmission();
  delay(300);

  encoder.attachHalfQuad(DT, CLK);
  encoder.setCount(0);
}
 
void loop()
{  
    if((Serial.available() > 0)){
      
      incomingByte = Serial.read();
      //Serial.println(incomingByte);
      if (!data_stream){
        if(incomingByte == '-'){
        //Serial.print("negative");
        sign = -1;
        movement = 0;
        }
        else{
        sign = 1;
        movement = incomingByte - '0';
          }
          data_stream = true;
          }
      else if (isDigit(incomingByte)){
        //Serial.print("adding: ");
        movement = movement*10 + int(incomingByte - '0');
        //Serial.println(movement);
        }
        }
       else if((data_stream)){
            //Serial.print("test: ");
            //Serial.println(movement*sign);
            data_stream = false;
            if(sign == -1){
              movemotor(HIGH,LOW,movement);
              }
              else{
            movemotor(LOW,HIGH,movement);}
        }
         
        
  
  delay(25);
}

void read_sensors() {
  unsigned int data[7];

  // Request 7 bytes of data
  Wire.requestFrom(Addr, 7);

  // Read 7 bytes of data
  // staus, xAccl lsb, xAccl msb, yAccl lsb, yAccl msb, zAccl lsb, zAccl msb
  if (Wire.available() == 7) {
    data[0] = Wire.read();
    data[1] = Wire.read();
    data[2] = Wire.read();
    data[3] = Wire.read();
    data[4] = Wire.read();
    data[5] = Wire.read();
    data[6] = Wire.read();
  }

  // Convert the data to 12-bits
  int xAccl = ((data[1] * 256) + data[2]) / 16;
  if (xAccl > 2047) {
    xAccl -= 4096;
  }

  int yAccl = ((data[3] * 256) + data[4]) / 16;
  if (yAccl > 2047) {
    yAccl -= 4096;
  }

  int zAccl = ((data[5] * 256) + data[6]) / 16;
  if (zAccl > 2047) {
    zAccl -= 4096;
  }
  // Output data to serial monitor
  Serial.print(xAccl);
  Serial.print(",");
  Serial.print(yAccl);
  Serial.print(",");
  Serial.print(zAccl);
  Serial.print(",");
  Serial.print(currentPosition);
  Serial.print(",");
  Serial.print(pwmValue);
  Serial.println(",");
}

void test_sequence(int targetPosition) {
  movemotorPWM(0, NONE);
  Serial.print("Starting test sequence.");
  read_sensors();
  Direction dir = LEFT;

  for (int i = 0; i < 256; i++) {
    movemotorPWM(i, LEFT);
    while (currentPosition < targetPosition) {
      currentPosition = encoder.getCount();
      read_sensors();
    }
    movemotorPWM(0, NONE);
    delay(500);
    movemotorPWM(100, RIGHT);
    while (currentPosition > 0) {
      currentPosition = encoder.getCount();
      read_sensors();
    }
    movemotorPWM(0, NONE);
    delay(500);
  }
}