
int nCircles = 75;
float[] xSeeds = new float[nCircles];
float[] ySeeds = new float[nCircles];
float[] speeds = new float[nCircles];
int[] xSizes = new int[nCircles];
int[] ySizes = new int[nCircles];

void setup() {
  background(0);
  size(2100, 900);
  
  for (int i = 0; i < nCircles; i++) {
    xSeeds[i] = random(-1000, 1000);
    ySeeds[i] = random(-1000, 1000);
    speeds[i] = random(-0.02, 0.02);
    xSizes[i] = int(random(50, 100));
    ySizes[i] = int(random(50, 100));
  }
}

void draw() {
  fill(255, 10);
  noStroke();
  rect(0, 0, width, height);
  //stroke(255);
  
  fill(150);
  float x;
  float y;
  for (int i = 0; i < nCircles; i++) {
    x = noise(xSeeds[i]) * width;
    y = noise(ySeeds[i]) * height;
    
    xSeeds[i] += speeds[i];
    ySeeds[i] += speeds[i];
    
    //ellipse(x, y, xSizes[i], ySizes[i]);
    ellipse(x, y, 64, 64);
  }
}