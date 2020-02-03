int outputPin = 3;
int input0 = A0;
// Make an Array to hold the values received
const int arraySize = 800;
int readings[arraySize];
int tBase = 1000;
//Time between readings
int readingPointer; 
#define BLUE 3

#define GREEN 4

#define RED 3

#define delayTime 50000

void setup() {
 // pinMode(outputPin, OUTPUT);
 // pinMode(input0,INPUT);
    pinMode(RED,OUTPUT);
    pinMode(GREEN,OUTPUT);
    Serial.begin(9600);
    TCCR1B = TCCR1B & 0b11111000 | 0x05;
   

}

void loop() {
 /* if(Serial.available()> 0){
    int out=Serial.read();
    //analogWrite(outputPin,out);

    // Analog readings to delay tBase
    for(readingPointer = 0; readingPointer < arraySize; readingPointer++){
      readings[readingPointer] = analogRead(input0);
      delayMicroseconds(tBase);
    }
   // Put array values to serial port.
    for(readingPointer = 0; readingPointer < arraySize; readingPointer++){
      Serial.write(highByte(readings[readingPointer]<<6));
      
    }
  } */

  if(Serial.available() > 0){
    delayMicroseconds(delayTime);
    int redIntensity = Serial.read();
    int greenIntensity = redIntensity/10;
    analogWrite(RED,redIntensity);
    analogWrite(GREEN,greenIntensity);


}

}
