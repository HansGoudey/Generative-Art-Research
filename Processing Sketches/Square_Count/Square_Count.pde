final int ITERATIONS = 100;

float MAX_DIAMETER = 50;
float MIN_DIAMETER = 20;

void setup() {
  size(200, 200); 
  noLoop();
  ellipseMode(CENTER);
  noStroke();
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {
    for (int nSquares = 1; nSquares <= 100; nSquares++) {
      background(random(0, 255));
      
      for (int square = 0; square <= nSquares; square++) {
        //println();
        //println(float(nTriangles) / 100.0);
        //println(lerp(MIN_DIAMETER, MAX_DIAMETER, nTriangles / 100));
        float diameter = random(0.5, 1.5) * lerp(MAX_DIAMETER, MIN_DIAMETER, float(nSquares) / 100.0);
        float x = random(20, width - 20);
        float y = random(20, height - 20);
        pushMatrix();
        stroke(random(0,255));
        fill(random(0,255));
        rotate(random(0, 2*PI));
        rect(x, y, diameter, diameter);
        popMatrix();
      }
      
      String fileName = String.format("Square Count/nms-n%02d-%03d.png", nSquares, iteration);
      save(fileName);
    }
    println(iteration + 1);
  }
}
