import java.util.*;

public class Point {
  int x;
  int y;
  
  public Point(int xin, int yin) {
    this.x = xin;
    this.y = yin;
  }
}


int sidePoints = 15;
int boundary = 25;

//Point[] points = new Point[(sidePoints - 2) * 4 + 4];
Vector <Point> points = new Vector <Point>();

void setup() {
  background(230);
  strokeWeight(2);
  size(7001, 7001);
  
  // TOP
  for (int i = 0; i < sidePoints - 1; i++) {
    int x = int(lerp(boundary, width - boundary, (float)i / (float)(sidePoints - 1)));
    int y = boundary;
    points.add(new Point(x, y));
  }
  // RIGHT
  for (int i = 0; i < sidePoints - 1; i++) {
    int x = width - boundary;
    int y = int(lerp(boundary, height - boundary, (float)i / (float)(sidePoints - 1)));
    points.add(new Point(x, y));
  }
  // BOTTOM
  for (int i = 0; i < sidePoints - 1; i++) {
    int x = int(lerp(width - boundary, boundary, (float)i / (float)(sidePoints - 1)));
    int y = height - boundary;
    points.add(new Point(x, y));
  }
  // LEFT
  for (int i = 0; i < sidePoints - 1; i++) {
    int x = boundary;
    int y = int(lerp(height - boundary, boundary, (float)i / (float)(sidePoints - 1)));
    points.add(new Point(x, y));
  }
 
}

void draw() {
  stroke(50);
  
  for (int i = 0; i < points.size(); i++) {
    ellipse(points.get(i).x, points.get(i).y, 2, 2);
    for (int j = 0; j < points.size(); j++) {
      int ix = points.get(i).x;
      int iy = points.get(i).y;
      int jx = points.get(j).x;
      int jy = points.get(j).y;
      
      if (ix == boundary && jx == boundary) continue;
      if (ix == width - boundary && jx == width - boundary) continue;
      if (iy == boundary && jy == boundary) continue;
      if (iy == width - boundary && jy == width - boundary) continue;
      line(ix, iy, jx, jy);
    }
  }
  
  save("output.png");
  stop();
}