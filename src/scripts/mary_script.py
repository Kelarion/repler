import numpy as np
import matplotlib.pyplot as plt

def square_axis(ax=None):
    if ax is None:
        ax = plt.gca()
    xmin = np.min([ax.get_ylim(), ax.get_xlim()])
    xmax =  np.max([ax.get_ylim(), ax.get_xlim()])
    newlims = [xmin, xmax]

    plt.axis('equal')
    plt.axis('square')
    ax.set_xlim(newlims)
    ax.set_ylim(newlims)

##########################################################################

## Write each set of coefficients (you can change these to whatever you want)
eq1 = np.array([1,1])
eq2 = np.array([-1,1])
eq3 = np.array([-1,1])
eq4 = np.array([-1,-1])

## Organise the equations as rows of a matrix
eqs = np.stack([eq1, eq2, eq3, eq4]) # [eq3, ...]

## Try playing with different coefficients, like
# eq1 = np.array([0.3,1])
# eq2 = np.array([-1,-0.4])
## Some sets of equations will have no solutions ... 

## define the "intercept", i.e. the b in a*w = b
b1 = 1
b2 = 1
b3 = -1
b4 = 1

## Organize the intercepts as an array of shape (num_eq, 1)
b = np.array([[b1], [b2], [b3], [b4]]) # [[b3], ...]

## Enumerate w vectors in a grid
grid_size = 100
w1, w2 = np.meshgrid(np.linspace(-2,2,grid_size), np.linspace(-2,2,grid_size))
w1 = w1.flatten()
w2 = w2.flatten()

## The w vectors are stored as the columns of a matrix
## which has shape (2, grid_size**2)
w = np.stack([w1, w2])

## To compute a dot product, use '@' 
dots = eqs @ w

## The above code does the same thing as:
# dots = []
# for eq in eqs:
#     dots.append(eq.dot(w))
# dots = np.stack(dots)

## Plot the equation coefficients, along with the feasible regions
feasible = np.all(dots <= b, axis=0)

for eq in eqs: # this plots the coefficients as arrows
    plt.quiver(eq[0], eq[1], angles='xy', scale_units='xy', scale=1, zorder=10)
plt.scatter(w[0], w[1], c=feasible, cmap='bwr', alpha=0.5, marker='.')
square_axis()


