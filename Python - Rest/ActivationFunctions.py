import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


def main():
    # Pfad zum speichern festlegen
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Plots/finished/activations/"
    # Zu plottende Activation Functions festlegen
    activation_functions = {
        "ReLU(x)": tf.nn.relu,
        "Leaky-ReLU(x)": tf.nn.leaky_relu,
        "ELU(x)": tf.nn.elu,
        "Sigmoid(x)": tf.nn.sigmoid,
        "tanh(x)": tf.nn.tanh,
        "Softmax(x)": tf.nn.softmax,
        "Softplus(x)": tf.nn.softplus,
        "SELU(x)": tf.nn.selu,
        "Exp(x)": tf.keras.activations.exponential
    }
    # Intervall festlegen
    x = np.linspace(-3, 3, num=100)
    # Funktionswerte berechnen
    y = dict()
    for function_name in activation_functions:
        y[function_name] = activation_functions[function_name](x)
        # Sachen plotten
        font = {"family": "normal", "size": 13}
        matplotlib.rc("font", **font)
        fig, ax = plt.subplots()
        ax.grid(True, color="lightgray")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\it{f}(x)$")
        ax.axhline(y=0, color='k', alpha=0.5)
        ax.axvline(x=0, color='k', alpha=0.5)
        """
        # set the x-spine (see below for more info on `set_position`)
        ax.spines['left'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()
        """
        ax.plot(x, y[function_name], label=function_name)
        ax.legend()
        if save_path:
            plt.savefig(save_path + "_" + function_name + ".pdf", format="pdf")
        plt.show()



if __name__ == "__main__":
    main()
