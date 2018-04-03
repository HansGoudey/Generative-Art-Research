final int ITERATIONS = 20;

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
      ArrayList<Circle> circles = createCircles(numCircles);
      background(random(0, 255));
      drawCircles(circles);
      String fileName = String.format("Circle Count/nms-n%02d-%03d.png", numCircles, iteration);
      save(fileName);
      println(String.format("%02.02f", (iteration*100 + numCircles)/(100.0 * ITERATIONS)));
    }
  }
}


ArrayList<Circle> createCircles(int count) {
  ArrayList<Circle> circles = new ArrayList<Circle>(count);
  while (circles.size() < count) {
    Circle c = new Circle(); 
    boolean match = false;
    for (Circle c2 : circles) {
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
      circles.add(c);
    }
  }

  return circles;
}


void drawCircles(ArrayList<Circle> circles) {
  for (Circle c : circles) {

    float fillColor;
    do {
      fillColor = random(0, 255);
    } while (abs(brightness(g.backgroundColor) - fillColor) < 10);
    fill(fillColor);

    float diameter = random(MIN_DIAMETER, c.maxRadius*2);
    ellipse(c.x, c.y, diameter, diameter);
  }
}


class Circle {
  float x;
  float y;
  float maxRadius = MAX_DIAMETER;

  Circle() {
    x = random(0, width);
    y = random(0, height);
  }


  float distanceFrom(Circle c) {
    return sqrt((c.x - x) * (c.x - x) + (c.y - y) * (c.y - y));
  }


  void addNeighbor(float distance) {
    maxRadius = min(maxRadius, distance);
  }
}