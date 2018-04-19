final int ITERATIONS = 20;

float MAX_DIAMETER = 50;
float MIN_DIAMETER = 25;

void setup() {
  size(200, 200); 
  noLoop();
  ellipseMode(CENTER);
  noStroke();
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {
    for (int nTriangles = 1; nTriangles <= 100; nTriangles++) {
      background(random(0, 255));
      
      for (int triangle = 0; triangle <= nTriangles; triangle++) {
        //println();
        //println(float(nTriangles) / 100.0);
        //println(lerp(MIN_DIAMETER, MAX_DIAMETER, nTriangles / 100));
        float diameter = random(0.5, 1.5) * lerp(MAX_DIAMETER, MIN_DIAMETER, float(nTriangles) / 100.0);
        pushMatrix();
        rotate(random(0, 2*PI));
        float x = random(20, width - 20);
        float y = random(20, height - 20);
        fill(random(0, 255));
        triangle(x, y, x + diameter, y, x + diameter, y + diameter);
        popMatrix();
      }
      
      String fileName = String.format("Triangle Count/nms-n%02d-%03d.png", nTriangles, iteration);
      save(fileName);
    }
    println(iteration + 1);
  }
}
