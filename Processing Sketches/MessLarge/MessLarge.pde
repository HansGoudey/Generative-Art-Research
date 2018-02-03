int nImages = 0;
final float mess = 10;
final int iterations = 50;

void setup() {
  size(1440, 1440);
  strokeWeight(width / 200);
  frameRate(1);
}

void draw() {
  clear();
  float x;
  float y;
  float otherX;
  float otherY;
  float rotation;
  float scale;
  int choice;
  
  for (int iteration = 1; iteration < iterations; iteration++) {  // Each set of 100 images
    for (int m = 0; m < 100; m++) {  // Each of the images
      background(random(70, 200), random(70, 200), random(70, 200));
      
      scale = 10 + 100 * (m/100);
      for(int i = 0; i < Math.pow(m, 1.7)/3 * mess + 2; i++) {  // Each primitive drawm
        
        x = random(0, width);
        y = random(0, height);
        otherX = random(x - 40, x + 40);
        otherY = random(y - 40, y + 40);
        rotation = random(0, TWO_PI);
        
        if (random(0,1) < (float)m/100.0 - 0.3) {
          noFill();
        }
        else {
          fill(random(50, 255), random(50, 255), random(50, 255));
        }
        stroke(random(0,100));
        
        pushMatrix();
        rotate(random(0, PI));
        choice = Math.round(random(1,7));  // Choice of which 2D primitive to draw
        //print("\nChoice: " + choice);
        switch(choice) {
          case 1:  // Arc
            arc(x, y, otherX, otherY, rotation, random(0, TWO_PI));
            break;
          case 2:  // Ellipse
            ellipse(x, y, random(40, 300), random(40, 300));
            break;
          case 3:  // Line
            line(x, y, otherX, otherY);
            break;
          case 4:  // Point
            point(x, y);
            break;
          case 5:  // Quad
            quad(x, y, otherX, otherY, random(x-scale, x+scale), random(y-scale, y+scale), random(x-scale, x+scale), random(y-scale, y+scale));
            break;
          case 6:  // Rect
            rect(x, y, random(50, 300), random(50, 300), random(0, 100));
            break;
          case 7: // Triangle
            triangle(x, y, random(x-200, x+200), random(y-300, y+300), random(x-300, x+300), random(y-300, y+300));
            break;
          default:
            print("Uh oh");
        }
        popMatrix();
      }
      save(String.format("Large/mes-m%02d-%03d.png", m, iteration));
    }
    print("Iteration finished\n");
  }
}