int count = 0;

void setup() {
  size(200, 200);
  //frameRate(4);
  noStroke();
}


void draw() {
  clear();
  background(random(0, 205));
  
  int roundness = 0;
  if (random(1) > 0.5) roundness = 99;
  
  for (int i = 0; i < random(1,5); i++) {
    int x = (int)random(width * 0.1, width * 0.9);
    int y = (int)random(height * 0.1, height * 0.9);
    
    float size = random(0.1 * (height + width)/2, 0.3 * (height + width)/2); 
    
    fill(random(50, 255));
    if (roundness == 0) rect(x, y, size, size);
    else ellipse(x, y, size, size);
  }
  
  String fname = String.format("img-r%02d-%03d.png", roundness, count);
  save(fname);
  count++;
  if (count > 1000) stop();
}