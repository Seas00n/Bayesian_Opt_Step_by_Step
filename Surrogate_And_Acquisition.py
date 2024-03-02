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
black_box_output = black_box_function(x_range) 
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(x_range, black_box_output, label='True Value')

# 采样的x
num_samples = 10
sample_x = np.random.choice(x_range, size=num_samples)

# 每个x的输出
sample_y = black_box_function(sample_x)
ax0.scatter(sample_x, sample_y, color= 'red', label="Sample Point")

# GPR with RBF kernel
kernel = RBF(length_scale=1.0)
gp_model = GaussianProcessRegressor(kernel=kernel)

# 用采样点拟合
gp_model.fit(sample_x.reshape(-1,1), sample_y)

# Generate predictions using the Gaussian process model
y_pred, y_std = gp_model.predict(x_range.reshape(-1, 1), return_std=True)

ax0.plot(x_range, y_pred, 'b--', label='GPR Predicted Value')
ax0.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='b',alpha=0.2)
ax0.legend()

# EI
def expected_improvement(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (y_pred - best_y) / y_std
    ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

best_idx = np.argmax(sample_y)
best_x = sample_x[best_idx]
best_y = sample_y[best_idx]

ei = expected_improvement(x_range, gp_model, best_y)
ax1 = fig.add_subplot(212)
ax1.plot(x_range, ei*10, label='EI')

# UCB
def upper_confidence_bound(x, gp_model, beta=2.0):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    ucb = y_pred + beta * y_std
    return ucb

ucb = upper_confidence_bound(x_range, gp_model)
ax1.plot(x_range, ucb, label='UCB')


# PI
def probability_of_improvement(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (y_pred - best_y) / y_std
    pi = norm.cdf(z)
    return pi
pi = probability_of_improvement(x_range, gp_model, best_y)
ax1.plot(x_range, pi*5, label='PI')

ax1.legend()


plt.show()