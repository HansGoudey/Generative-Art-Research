public class Point {
  int x;
  int y;
  float direction;
  int lastX;
  int lastY;
  float lastDirection;

  Point(int x, int y, float direction) {
    setPosition(x, y);
    this.direction = direction;
    this.lastX = x;
    this.lastY = y;
    this.lastDirection = direction;
  }

  void setPosition(int x, int y) {
    this.lastX = this.x;
    this.lastY = this.y;
    this.x = x;
    this.y = y;
  }

  void changePosition(int dx, int dy) {
    this.lastX = this.x;
    this.lastY = this.y;
    this.x += dx;
    this.y += dy;
  }

  void changeDirection(float directionChange) {
    this.lastDirection = this.direction;
    direction = direction % 360;
    this.direction += directionChange;
  }

  void move(float distance) {
    changePosition(Math.round(distance * cos(radians(this.direction))), Math.round(distance * sin(radians(this.direction))));
  }
}

int nPoints = 200;

void setup() {
  size(1000, 1000);
  frameRate(1);
}


void draw() {
  clear();
  int point = 0;
  
  stroke(255);
  strokeWeight(4);

  line(0, height/2, width, height/2);

  while (point < nPoints) {
    Point dot = new Point(Math.round(random(0, width)), Math.round(random(0, height)), random(0, 360));
    
    while (point < nPoints) {
      //fill(255);
      //ellipse(newDot.x, newDot.y, 4, 4);

      //print("dot direction is: " + dot.direction + "\n");
      dot.move(10);
      
      //print("noise from " + point + " is " + noise(point) + "\n");
      float directionChange = pow((float)(10 * (noise(point*0.01) - 0.5)), 1.0);
      //print("directionChange: " + directionChange + "\n");
      //fill(204, 102, 0);
      //ellipse(point, directionChange*50 + height/2, 2, 2);
      
      print("On Draw dot location: (" + dot.lastX + ", " + dot.lastY + ")  newDot location: (" + dot.x + ", " + dot.y + ")\n");
      line(dot.lastX, dot.lastY, dot.x, dot.y);
     
      dot.changeDirection(directionChange); 

      point++;

      if (dot.x > width || dot.x < 0 || dot.y > height || dot.y < 0) break;
    }
  }
  //stop();
}


void mouseClicked() {
  save("lines" + random(0,4000) + ".png");
}