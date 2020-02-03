int wait = 200;
void lA () {dot();dash();ss();}//letter A in morse code!
void lB () {dash();dot();dot();dot();ss();}//same for B
void lC () {dash();dot();dash();dot();ss();}
void lD () {dash();dot();dot();ss();}
void lE () {dot();ss();}
void lF () {dot();dot();dash();dot();ss();}
void lG () {dash();dash();dot();ss();}
void lH () {dot();dot();dot();dot();ss();}
void lI () {dot();dot();ss();}
void lJ () {dot();dash();dash();dash();ss();}
void lK () {dash();dot();dash();ss();}
void lL () {dot();dash();dot();dot();ss();}
void lM () {dash();dash();ss();}
void lN () {dash();dot();ss();}
void lO () {dash();dash();dash();ss();}
void lP () {dot();dash();dash();dot();ss();}
void lQ () {dash();dash();dot();dash();ss();}
void lR () {dot();dash();dot();ss();}
void lS () {dot();dot();dot();ss();}
void lT () {dash();ss();}
void lU () {dot();dot();dash();ss();}
void lV () {dot();dot();dot();dash();ss();}
void lW () {dot();dash();dash();ss();}
void lX () {dash();dot();dot();dash();ss();}
void lY () {dash();dot();dash();dash();ss();}
void lZ () {dash();dash();dot();dot();ss();}
void n1 () {dot();dash();dash();dash();dash();ss();}//number 1 in morse code
void n2 () {dot();dot();dash();dash();dash();ss();}
void n3 () {dot();dot();dot();dash();dash();ss();}
void n4 () {dot();dot();dot();dot();dash();ss();}
void n5 () {dot();dot();dot();dot();dot();ss();}
void n6 () {dash();dot();dot();dot();dot();ss();}
void n7 () {dash();dash();dot();dot();dot();ss();}
void n8 () {dash();dash();dash();dot();dot();ss();}
void n9 () {dash();dash();dash();dash();dot();ss();}
void n0 () {dash();dash();dash();dash();dash();ss();}
void space () {delay (7*wait);}//space between words
void dot () {digitalWrite(13,HIGH); delay (wait); digitalWrite(13,LOW); delay (wait);}
void dash () {digitalWrite(13,HIGH); delay (wait*3); digitalWrite(13,LOW); delay (wait);}
void ss () {delay(wait*3);}//space between letters

void setup() {
  // put your setup code here, to run once:
  pinMode(13,OUTPUT);
  Serial.begin(9600);
  
}


void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()){
    char input = Serial.read();
    if (input == 'a') {lA();}//if the input is a or A go to function lA
    if (input == 'b') {lB();}//same but with b letter
    if (input == 'c') {lC();}
    if (input == 'd') {lD();}
    if (input == 'e') {lE();}
    if (input == 'f') {lF();}
    if (input == 'g') {lG();}
    if (input == 'h') {lH();}
    if (input == 'i') {lI();}
    if (input == 'j') {lJ();}
    if (input == 'k') {lK();}
    if (input == 'l') {lL();}
    if (input == 'm') {lM();}
    if (input == 'n') {lN();}
    if (input == 'o') {lO();}
    if (input == 'p') {lP();}
    if (input == 'q') {lQ();}
    if (input == 'r') {lR();}
    if (input == 's') {lS();}
    if (input == 't') {lT();}
    if (input == 'u') {lU();}
    if (input == 'v') {lV();}
    if (input == 'w') {lW();}
    if (input == 'x') {lX();}
    if (input == 'y') {lY();}
    if (input == 'z') {lZ();}
  }
  
  

}
