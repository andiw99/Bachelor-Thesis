from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm


def potential(x,y):
    return  0.8 * x ** 4 + 0.8 * y ** 4 - x ** 2 - y ** 2 - 2 * x - (x - y) ** 2 + 5


def pot2(x,y):
    return 0.3 * x ** 4 + 0.3 * y ** 4 - x + y - 1.5 * x ** 2 + 2


def main():
    X = np.linspace(-2, 2, 50)
    Y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(X, Y)
    Z = pot2(X,Y)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0, 10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("C(w,b)")
    plt.show()

    X_pos = np.linspace(0, 2, 25)
    Y_pos = np.linspace(0, 2, 25)
    Y_neg = np.linspace(-2, 0, 25)
    X_neg = np.linspace(-2, 0, 25)
    X = np.linspace(-2, 2, 50)
    X_x_y_pos, Y_x_y_pos = np.meshgrid(X, Y_pos)
    Z_x_y_pos = pot2(X_x_y_pos, Y_x_y_pos)

    X_x_neg_y_neg, Y_x_neg_y_neg = np.meshgrid(X_neg, Y_neg)
    Z_x_neg_y_neg = pot2(X_x_neg_y_neg, Y_x_neg_y_neg)

    X_x_pos_y_neg, Y_x_pos_y_neg = np.meshgrid(X_pos, Y_neg)
    Z_x_pos_y_neg = pot2(X_x_pos_y_neg, Y_x_pos_y_neg)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_x_y_pos, Y_x_y_pos, Z_x_y_pos, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(X_x_neg_y_neg, Y_x_neg_y_neg, Z_x_neg_y_neg, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(X_x_pos_y_neg, Y_x_pos_y_neg, Z_x_pos_y_neg, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=1)
    ax.set_zlim(0, 10)
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("C(w,b)")
    plt.show()

if __name__ == "__main__":
    main()

