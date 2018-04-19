final int ITERATIONS = 100;

float MAX_DIAMETER = 50;
float MIN_DIAMETER = 20;

void setup() {
  size(200, 200); 
  noLoop();
  ellipseMode(CENTER);
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {
    for (int nLines = 1; nLines <= 100; nLines++) {
      background(random(101, 255));
      
      for (int line = 0; line <= nLines; line++) {
        //println();
        //println(float(nTriangles) / 100.0);
        //println(lerp(MIN_DIAMETER, MAX_DIAMETER, nTriangles / 100));
        float diameter = random(0.5, 1.5) * lerp(MAX_DIAMETER, MIN_DIAMETER, float(nLines) / 100.0);
        float x = random(25, width - 25);
        float y = random(25, height - 25);
        pushMatrix();
        rotate(random(0, 2*PI));
        stroke(random(0, 99));
        line(x - diameter, y, x + diameter, y);
        popMatrix();
      }
      
      String fileName = String.format("Line Count/nms-n%02d-%03d.png", nLines, iteration);
      save(fileName);
    }
    println(iteration + 1);
  }
}
