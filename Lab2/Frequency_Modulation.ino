int val = 140;
void setup() {
  // put your setup code here, to run once:
  //TCCR2B = TCCR2B & 0b11111000 | 0x02;
  //TCCR1B = TCCR1B & 0b11111000 | 0x02;
  //TCCR2B = TCCR2B & 0b11111000 | 0x06;
  //TCCR1B = TCCR1B & 0b11111000 | 0x04;
  
  pinMode(3, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10,OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  analogWrite(3, val);
  analogWrite(9, val);
  analogWrite(10,val);

}
