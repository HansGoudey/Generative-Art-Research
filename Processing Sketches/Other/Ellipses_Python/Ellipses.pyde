
n_circles = 100;

float x_starts = [random.uniform(-1000, 1000) for x in range(n_circles)]

def setup():
    background(0)
    size(1050,450)

def draw():
    for i in range(n_circles):
        ellipse(x_starts[i], height/2, 64, 64)
    