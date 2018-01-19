int cnt = 0;

void setup() {
  size(200, 200);
  frameRate(10);
}

void randrect(float rad) {
  float ww = width;
  float hh = height;
  float r = random(180);
  stroke(r);
  fill(r);
  float x = random(-ww, ww);
  float y = random(-hh, hh);
  float w = random(ww/8, ww/2);
  float h = random(hh/8, hh/2);
  float a = random(PI);
  float m = min(w, h);
  pushMatrix();
  translate(x, y);
  rotate(a);
  rect(0, 0, w, h, rad * m);
  popMatrix();
}

void draw() {
  clear();
  background(255);
  //float rad = random(1);
  int iter = cnt / 100;
  int r = cnt % 100;
  float rad = r / 100.0;
  for (int i=0; i<50; i++) {
    randrect(rad);
  }
  String fname = String.format("img-r%02d-%03d.png", r, iter);
  save(fname);
  cnt++;
}