int nImages = 0;
final float mess = 4;
final int iterations = 50;

void setup() {
  size(200, 200);
  frameRate(1);
}

void draw() {
  clear();
  float x;
  float y;
  float otherX;
  float otherY;
  float rotation;
  int choice;
  
  for (int iteration = 0; iteration < iterations; iteration++) {  // Each set of 100 images
    for (int m = 0; m < 100; m++) {  // Each of the images
      background(random(70, 240));
      
      
      for(int i = 0; i < m * mess + 2; i++) {  // Each primitive drawm
        
        x = random(0, width);
        y = random(0, height);
        otherX = random(x - 50, x + 50);
        otherY = random(y - 50, y + 50);
        rotation = random(0, TWO_PI);
        
        if (random(0,1) < (float)m / 100.0 - 0.2) {
          noFill();
        }
        else {
          fill(random(50, 255));
        }
        stroke(random(0,150));
        
        choice = (int)random(1,7);  // Choice of which 2D primitive to draw
        //print("\nChoice: " + choice);
        switch(choice) {
          case 1:  // Arc
            arc(x, y, otherX, otherY, 0, rotation);
            break;
          case 2:  // Ellipse
            ellipse(x, y, random(5, 50), random(5, 50));
            break;
          case 3:  // Line
            line(x, y, otherX, otherY);
            break;
          case 4:  // Point
            point(x, y);
            break;
          case 5:  // Quad
            quad(x, y, otherX, otherY, random(x-40, x+40), random(y-40, y+40), random(x-40, x+40), random(y-40, y+40));
            break;
          case 6:  // Rect
            rect(x, y, random(15, 25), random(15, 25));
            break;
          case 7: // Triangle
            triangle(x, y, otherX, otherY, random(x-40, x+40), random(y-40, y+40));
            break;
          default:
            print("Uh oh");
        }
      }
      stop();
      save(String.format("Output/mes-f%02d-%03d.png", m, iteration));
    }
    print("Iteration finished\n");
  }
}