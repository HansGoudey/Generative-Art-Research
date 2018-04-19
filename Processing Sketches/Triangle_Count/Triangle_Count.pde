final int ITERATIONS = 100;

final float MIN_DISTANCE = 4;
final float MIN_DIAMETER = 10;
final float MAX_DIAMETER = 50;



void setup() {
  size(200, 200); 
  noLoop();
  ellipseMode(CENTER);
  noStroke();
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {
    for (int numCircles = 1; numCircles <= 100; numCircles+=1) {
      ArrayList<Triangle> triangles = createTriangles(numCircles);
      background(random(0, 255));
      drawTriangles(triangles);
      String fileName = String.format("Triangle Count/nms-n%02d-%03d.png", numCircles, iteration);
      save(fileName);
      println(String.format("%02.02f", (iteration*100 + numCircles)/(100.0 * ITERATIONS)));
    }
  }
}


ArrayList<Triangle> createTriangles(int count) {
  ArrayList<Triangle> triangles = new ArrayList<Triangle>(count);
  while (triangles.size() < count) {
    Triangle c = new Triangle(); 
    boolean match = false;
    for (Triangle c2 : triangles) {
      float distance = c.distanceFrom(c2);
      if (distance < MIN_DISTANCE) {
        match = true;
        break;
      } else {
        c.addNeighbor(distance);
        c2.addNeighbor(distance);
      }
    }
    if (! match) {
      triangles.add(c);
    }
  }

  return triangles;
}


void drawTriangles(ArrayList<Triangle> circles) {
  for (Triangle c : circles) {

    float fillColor;
    do {
      fillColor = random(0, 255);
    } while (abs(brightness(g.backgroundColor) - fillColor) < 10);
    fill(fillColor);

    float diameter = random(MIN_DIAMETER, c.maxRadius*2);
    pushMatrix();
    rotate(random(0, 2*PI));
    triangle(c.x, c.y, c.x + diameter, c.y, c.x + diameter, c.y + diameter);
    popMatrix();
  }
}


class Triangle {
  float x;
  float y;
  float maxRadius = MAX_DIAMETER;

  Triangle() {
    x = random(0, width);
    y = random(0, height);
  }


  float distanceFrom(Triangle c) {
    return sqrt((c.x - x) * (c.x - x) + (c.y - y) * (c.y - y));
  }


  void addNeighbor(float distance) {
    maxRadius = min(maxRadius, distance);
  }
}
