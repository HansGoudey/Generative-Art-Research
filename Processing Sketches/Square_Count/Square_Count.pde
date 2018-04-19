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
      ArrayList<Square> squares = createSquares(numCircles);
      background(random(0, 255));
      drawSquares(squares);
      String fileName = String.format("Square Count/nms-n%02d-%03d.png", numCircles, iteration);
      save(fileName);
      println(String.format("%02.02f", (iteration*100 + numCircles)/(100.0 * ITERATIONS)));
    }
  }
}


ArrayList<Square> createSquares(int count) {
  ArrayList<Square> squares = new ArrayList<Square>(count);
  while (squares.size() < count) {
    Square c = new Square(); 
    boolean match = false;
    for (Square c2 : squares) {
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
      squares.add(c);
    }
  }

  return squares;
}


void drawSquares(ArrayList<Square> squares) {
  for (Square c : squares) {

    float fillColor;
    do {
      fillColor = random(0, 255);
    } while (abs(brightness(g.backgroundColor) - fillColor) < 10);
    fill(fillColor);

    float diameter = random(MIN_DIAMETER, c.maxRadius*2);
    pushMatrix();
    rotate(random(0, 2*PI));
    rect(c.x, c.y, diameter, diameter);
    popMatrix();
  }
}


class Square {
  float x;
  float y;
  float maxRadius = MAX_DIAMETER;

  Square() {
    x = random(0, width);
    y = random(0, height);
  }


  float distanceFrom(Square c) {
    return sqrt((c.x - x) * (c.x - x) + (c.y - y) * (c.y - y));
  }


  void addNeighbor(float distance) {
    maxRadius = min(maxRadius, distance);
  }
}
