import NumpyTests
import generator
from pattern import Checker, Circle, Spectrum

def main():
    checker = Checker(12,3)
    checker.draw()
    checker.show()

    circle = Circle(12,3,(4,4))
    circle.draw()
    circle.show()

    spectrum = Spectrum(12)
    spectrum.draw()
    spectrum.show()

    imagegenerator = generator.ImageGenerator('./exercise_data/', './Labels.json', 12,(50, 50,3))
    imagegenerator.show()

if __name__ == "__main__":
    main()
