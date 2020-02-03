int readings[500];
int t = 1000;
void setup() {
  // put your setup code here, to run once:
  pinMode(9, OUTPUT);
  pinMode(10,OUTPUT);
  pinMode(A0, INPUT);
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:

  if(Serial.available() > 0){
    int out = Serial.read();
    analogWrite(9,out);
    delayMicroseconds(50000000);
    for (int i = 0; i < 800; i++){
      readings[i] = analogRead(A0);
      delayMicroseconds(5000);
    }
    for (int j = 0; j < 800; j++){
      Serial.write(highByte(readings[j] << 6));
    }
  }

}
