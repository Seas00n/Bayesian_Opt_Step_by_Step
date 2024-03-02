import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

np.random.seed(42)

def  black_box_function ( x ): 
    y = np.sin(x) + np.cos( 2 *x) 
    return y 

# x 值的范围
x_range = np.linspace(- 2 *np.pi, 2 *np.pi, 100 ) 

# 每个 x 值的输出
true_value = black_box_function(x_range) 

# 采样
num_samples = 5
sample_x = np.random.choice(x_range, size=num_samples)
sample_y = black_box_function(sample_x)

fig = plt.figure()

# UCB
def upper_confidence_bound(x, gp_model, beta=2.0):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    ucb = y_pred + beta * y_std
    return ucb

kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel)

for i in range(10):

    gp_model.fit(sample_x.reshape(-1, 1), sample_y)
    y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)
    

    # policy search
    ucb = upper_confidence_bound(x_range, gp_model)

    if i % 2 == 0:
        ax = fig.add_subplot(5,1,(i/2)+1)
        ax.plot(x_range,true_value,'black')
        ax.scatter(sample_x, sample_y, color='r')
        ax.plot(x_range, y_pred,'b')
        ax.plot(x_range, ucb, 'r--')
    
    new_x = x_range[np.argmax(ucb)]
    new_y = black_box_function(new_x)
    sample_x = np.append(sample_x, new_x)
    sample_y = np.append(sample_y, new_y)

plt.show()