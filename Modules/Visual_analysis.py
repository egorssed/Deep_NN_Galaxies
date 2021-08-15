import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

image_size=64
x = np.linspace(0, 64, 64)
y = np.linspace(0, 64, 64)
X, Y = np.meshgrid(x, y)
xdata = np.vstack((X.ravel(), Y.ravel()))

def plot_galaxies(*args,dimensions='2d',colorbar=False):
    #(num_of_rows,num_of_cols,im_shape[0],im_shape[1])
    args = [x.squeeze() for x in args]
    #num of cols
    n = min([x.shape[0] for x in args])
    I_max=np.max(args)
    print('Maximal brightness',I_max)

    if dimensions=='2d':
      fig=plt.figure(figsize=(2*n, 2*len(args)))
    else:
      fig=plt.figure(figsize=(5*n, 5*len(args)))

    for row in range(len(args)):
        for col in range(n):
            if dimensions=='2d':
                ax = fig.add_subplot(len(args),n,row*n+col+1)
                ax.imshow(args[row][col].squeeze(),cmap='Greys_r',vmax=I_max)
            else:
                ax = fig.add_subplot(len(args),n,row*n+col+1, projection='3d')
                ax.plot_surface(X, Y, args[row][col].squeeze(), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
                ax.set_zlim(0,I_max)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.show()







