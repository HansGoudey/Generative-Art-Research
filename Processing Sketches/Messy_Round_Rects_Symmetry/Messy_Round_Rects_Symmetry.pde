int cnt = 0;

void setup() {
  size(200, 200);
  frameRate(2);
}

void randrect(float rad) {  
  float r = random(180);
  stroke(r);
  fill(r);
  float x = random(0, width);
  float y = random(0, height);
  float w = random(width/8, width/3);
  float h = random(height/8, height/3);
  float a = random(PI);
  float m = min(w, h) / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, h, rad * m);
  popMatrix();
}

void draw() {
  for (int round = 0; round < 100; round++) {
    for (int messy = 0; messy < 100; messy++) {
      clear();
      background(random(0, 255));
  
      float rad = round / 100.0;
      for (int i=0; i < Math.round(messy/2); i++) {
        randrect(rad);
      }
      
      cnt++;
      print(cnt + "\n");
      String fname = String.format("Output/rmr-r%02d-m%02d-%03d.png", round, messy, cnt % 1000);
      //save(fname);
    }
  }
  stop();
}