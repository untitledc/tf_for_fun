import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

cmap = plt.cm.coolwarm
X = np.array([[1.0, 0.8, 2.5, 0.2, 0.0, -0.5, -2.0],
              [1.0, 0.8, 2.5, 0.2, 0.0, -0.5, -2.0],
              [1.0, 0.8, 2.5, 0.2, 0.0, -0.5, -2.0],
              [1.0, 0.8, 2.5, 0.2, 0.0, -0.5, -2.0],
              [1.0, 0.8, 2.5, 0.2, 0.0, -0.5, -2.0]
              ])
abs_max = np.max(np.abs(X))
print(abs_max)

color_norm = matplotlib.colors.Normalize(-abs_max, abs_max)
img = plt.imshow(X, cmap=cmap, norm=color_norm)
plt.colorbar(img, cmap=cmap, norm=color_norm)
plt.savefig('try_plot.png')
plt.title('test')
plt.close()