import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_boundary(model, points):
    RANGE = 0.55
    x_mesh = np.arange(-RANGE, RANGE, 0.001)
    y_mesh = np.arange(-RANGE, RANGE, 0.001)
    grid_x, grid_y = np.meshgrid(x_mesh, y_mesh)
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]

    classfications = model.predict(grid).argmax(axis=1)
    classfications_grid = classfications.reshape(grid_x.shape)

    BLUE_AND_GREEN = ListedColormap(['#BBBBFF', '#BBFFBB'])
    plt.contourf(grid_x, grid_y, classfications_grid, cmap=BLUE_AND_GREEN)

def plot_data_by_label(input_variables, labels, label_selector, symbol):
    points = input_variables[labels.flatten() == label_selector]
    plt.plot(points[:, 0], points[:, 1], symbol, markersize=4)

def show(model, x, y, title="Decision Boundary"):
    plot_boundary(model, x)
    plot_data_by_label(x, y, 0, 'bs')
    plot_data_by_label(x, y, 1, 'gs')
    plt.title(title)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.show()



