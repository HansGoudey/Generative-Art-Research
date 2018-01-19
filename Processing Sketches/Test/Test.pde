float xoff = 0.0;
float dx = 0.01;
float yoff = 3456.9;
float dy = 0.01;


void setup() {
  background(0);
  size(1000, 1000);
}

void draw() {
  fill(0, 10);
  noStroke();
  rect(0, 0, width, height);
  stroke(255);
  
  float x = noise(xoff) * width;
  float y = noise(yoff) * height;
  
  xoff += dx;
  yoff -= dy;
  
  //ellipse(500, y, 64, 64);
  //ellipse(x, 500, 64, 64);
  ellipse(x, y, 50, 50);
}