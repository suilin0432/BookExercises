import matplotlib.pyplot as plt
import math

class Exercise13_3(object):
    def __init__(self):
        pass

    def draw(self, ax, x, y, title, color="blue"):
        curve = ax.plot(x, y, color)
        if title:
            plt.title(title)
        return curve


    def run(self):
        # range of x
        x = [0.01 * i for i in range(1, 100)]
        # Beta(0.5, 0.5)
        y1 = [1/math.pi*math.pow(i, -0.5)*pow((1-i), -0.5) for i in x]
        ax1 = plt.subplot(2,2,1)
        self.draw(ax1, x, y1, "Beta(0.5, 0.5)")
        # Beta(1, 5)
        y2 = [5*math.pow(1-i, 4) for i in x]
        ax2 = plt.subplot(2,2,2)
        self.draw(ax2, x, y2, "Beta(1, 4)")
        # Beta(2, 2)
        y3 = [6*i*(1-i) for i in x]
        ax3 = plt.subplot(2,2,3)
        self.draw(ax3, x, y3, "Beta(2, 2)")
        # show all the curve
        ax4 = plt.subplot(2,2,4)
        curve1, = self.draw(ax4, x, y1, "Compare the curves", "blue")
        curve2, = self.draw(ax4, x, y2, "", "red")
        curve3, = self.draw(ax4, x, y3, "", "black")
        plt.legend([curve1, curve2, curve3], ["Beta(0.5, 0.5)", "Beta(1, 4)", "Beta(2, 2)"], loc = 'upper right')
        plt.show()




obj = Exercise13_3()
obj.run()

