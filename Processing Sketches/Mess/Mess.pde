int count = 0;
final float mess = 4;
final int iterations = 20;

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
  float scale;
  int choice;
  
  for (int iteration = 0; iteration < iterations; iteration++) {  // Each set of 100 images
    for (int m = 0; m < 100; m++) {  // Each of the images
      //print(count + "\n");
      background(random(70, 240));
      if (count >= 2000) {
        print("Stopping at " + count + " images");
        stop();
      }
      count++;
      scale = 10 + 40 * (m/100);
      for(int i = 0; i < m * mess + 2; i++) {  // Each primitive drawm
        
        x = random(0, width);
        y = random(0, height);
        otherX = random(x - 50, x + 50);
        otherY = random(y - 50, y + 50);
        rotation = random(0, TWO_PI);
        
        if (random(0,1) < (float)m/100.0 - 0.2) {
          noFill();
        }
        else {
          fill(random(50, 255));
        }
        stroke(random(0,150));
        
        pushMatrix();
        rotate((m/100) * random(-PI, PI));
        strokeWeight(1);
        choice = (int)random(1,7);  // Choice of which 2D primitive to draw
        //print("\nChoice: " + choice);
        //print("Made it to primitive choice, image #" + count);
        switch(choice) {
          case 1:  // Arc
            arc(x, y, otherX*0.75, otherY*0.75, random(0, TWO_PI), rotation);
            break;
          case 2:  // Ellipse
            ellipse(x, y, random(5, scale), random(5, scale));
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
            rect(x, y, random(15, scale*1.5), random(15, scale*1.5), random(0, 15));
            break;
          case 7: // Triangle
            triangle(x, y, otherX, otherY, random(x-30, x+30), random(y-30, y+30));
            break;
          default:
            print("Uh oh");
        }
        popMatrix();
        strokeWeight(2);
        stroke(random(0,50), 20);
        for (int point = 0; point < m; point++) {
          point(random(0,width), random(0,height));
        }
      }
      save(String.format("Output/mes-m%02d-%03d.png", m, iteration));
    }
    print(count);
  }
}