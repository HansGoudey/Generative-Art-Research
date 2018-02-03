final int ITERATIONS = 30;
final int ANT_ITERATIONS = 700;
final int NUM_ANTS = 20000;
ArrayList<Ant> ants = new ArrayList<Ant>(NUM_ANTS);

void setup() {
  size(1440, 1440);

  for (int i =0; i <NUM_ANTS; i++) {
    ants.add(new Ant());
  }

  background(255);
  stroke(255, 10);
  noLoop();
}


void draw() {
  for (int iteration = 0; iteration < ITERATIONS; iteration++) {  // Every 100 images
    for (int i = 0; i <= 99; i+=1) {  // Every image
      background(0);
      noiseSeed(System.currentTimeMillis());
      noiseDetail(6, i/100.0 + .001); // we need the .001 to mkae sure this doesn't go to zero
      for (Ant ant : ants) {
        for (int z = 0; z < ANT_ITERATIONS; z++) {
          ant.draw(.005);
        }
      }
      String fileName = String.format("samples/ant-r%02d-%03d.png", 100 - i, iteration);
      save(fileName);
      println(String.format("%02.02f", (iteration*100 + i)/(100.0 * ITERATIONS)));
    }
  }
}




class Ant {
  float x;
  float y;
  float heading;
  int r;
  int g;
  int b;

  Ant() {
    randomPosition();
    r = Math.round(noise((x + y) * 0.005) * 255);
    g = Math.round(noise((x + y) * 0.005 + 100) * 255);
    b = Math.round(noise((x + y) * 0.005 + 200) * 255);
  }

  void draw(float noiseScale) {
    float newX, newY;

    heading = noise(x*noiseScale, y*noiseScale)*TWO_PI;

    newX = x + sin(heading);
    newY = y +  cos(heading);
    stroke(r, g, b, 35);
    line(x, y, newX, newY);

    x = newX;
    y = newY;

    if (x < 0 | x > width) {
      randomPosition();
    }
    if (y < 0 | y > height) {
      randomPosition();
    }
  }

  void randomPosition() {
    x = random(0, width);
    y = random(0, height);
  }
}