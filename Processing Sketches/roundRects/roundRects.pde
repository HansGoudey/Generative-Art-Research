int cnt = 0;

void setup() {
  size(224, 224);
  //frameRate(2);
}

void randrect(float rad) {  
  float r = random(180);
  stroke(r);
  fill(r);
  float x = random(0, width);
  float y = random(0, height);
  float w = random(width/8, width/2);
  float h = random(height/8, height/2);
  float a = random(PI);
  float m = min(w, h) / 2;
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, h, rad * m);
  popMatrix();
}

void draw() {
  clear();
  background(random(0, 255));
  //float rad = random(1);
  int iter = cnt / 100;
  int r = cnt % 100;
  float rad = r / 100.0;
  for (int i=0; i<random(2,25); i++) {
    randrect(rad);
  }
  String fname = String.format("RoundRects224/rdr-r%02d-%03d.png", r, iter);
  save(fname);
  cnt++;
  if (cnt > 10000) stop();
}