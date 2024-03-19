import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm  
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

fig = plt.figure(figsize=(14,7))
grid = plt.GridSpec(1,3,wspace=0.5, hspace=0.5)
ax0 = plt.subplot(grid[0,0:2],projection='3d')
ax1 = plt.subplot(grid[0,2])

def black_box_cost(X,Y):
    return 0.3*np.sin(X*20/np.pi)+0.3*np.cos(Y*20/np.pi) 

x_range = np.arange(0,1,0.01)
y_range = np.arange(0,1,0.01)
X, Y = np.meshgrid(x_range, y_range)
Z = black_box_cost(X,Y)
ax0.plot_surface(X,Y,Z,linewidth=0,cmap=cm.coolwarm,
                 norm=matplotlib.colors.Normalize(vmin=-0.7, vmax=1),
                 alpha=0.4)
ax0.view_init(elev=19, azim=-115)
ax0.set_xlim([0,1])
ax0.set_ylim([0,1])
ax0.set_zlim([-1,1])
ax0.set_xlabel('x',fontsize=10, 
                   fontdict={"weight":'bold'})
ax0.set_ylabel('y',fontsize=10, 
                   fontdict={"weight":'bold'})
ax0.set_zlabel('z',fontsize=10, 
                   fontdict={"weight":'bold'})



ax1.imshow(Z, cmap=cm.coolwarm,norm=matplotlib.colors.Normalize(vmin=-0.7, vmax=1))
ax1.set_xticks([0,20,40,60,80,100])
ax1.set_yticks([0,20,40,60,80,100])
ax1.set_xticklabels([0,0.2,0.4,0.6,0.8,1])
ax1.set_yticklabels([0,0.2,0.4,0.6,0.8,1])
ax1.invert_yaxis()





num_samples = 10
theta = np.linspace(np.pi*2, np.pi*5, num_samples)
sample_x = np.round(0.025*np.sin(theta)*theta+0.5, 1).reshape((-1,))[::-1]
sample_y = np.round(0.025*np.cos(theta)*theta+0.5, 1).reshape((-1,))[::-1]
cost_z = black_box_cost(sample_x, sample_y).reshape((-1,1))
all_sample_3d = ax0.scatter(sample_x, sample_y, cost_z, c='r', linewidths=5)
new_sample_3d = ax0.scatter(sample_x[-1], sample_y[-1], cost_z[-1], c='m', marker='o', linewidths=8)
all_sample_2d = ax1.scatter(sample_x*100, sample_y*100, c='r', linewidths=2)
new_sample_2d = ax1.scatter(sample_x[-1]*100, sample_y[-1]*100, c='m',marker='o',linewidths=4)
all_sample_traj = ax1.plot(sample_x*100, sample_y*100,'r--')[0]

kernel = 0.2*RBF(length_scale=1, length_scale_bounds=(0.1, 10))
gp_model = GaussianProcessRegressor(kernel=kernel)
plt.show(block=False)
state_space = np.hstack([X.reshape(-1,1),Y.reshape(-1,1)])

def upper_confidence_bound(part_of_state_space, gp_model, beta=2.0):
    pred, std = gp_model.predict(part_of_state_space, return_std=True)
    ucb = pred+beta*std
    return ucb

gp_pred_surf = ax0.plot_surface(X,Y,np.zeros_like(Z),linewidth=0,cmap='rainbow',
                 norm=matplotlib.colors.Normalize(vmin=-0.7, vmax=1),
                 alpha=0.5)

for i in range(10):
    if i > 0:
        offset = np.hstack([sample_x.reshape(-1,1),sample_y.reshape(-1,1),cost_z.reshape(-1,1)])
        all_sample_3d._offsets3d = (sample_x[:-1], sample_y[:-1], cost_z[:-1])
        new_sample_3d._offsets3d = ([sample_x[-1]],[sample_y[-1]],[cost_z[-1]])
        all_sample_2d.set_offsets(offset[:,0:-1]*100)
        new_sample_2d.set_offsets([sample_x[-1]*100,sample_y[-1]*100])
        all_sample_traj.set_xdata(sample_x*100)
        all_sample_traj.set_ydata(sample_y*100)
        plt.show(block=False)
    part_of_state_space = np.hstack([sample_x.reshape(-1,1), sample_y.reshape(-1,1)])
    gp_model.fit(part_of_state_space, cost_z)
    cost_z_pred = gp_model.predict(state_space).reshape((100,100))
    #print(accuracy)
    gp_pred_surf.remove()
    gp_pred_surf = ax0.plot_surface(X,Y,cost_z_pred,linewidth=0,cmap='rainbow',
                 norm=matplotlib.colors.Normalize(vmin=-0.7, vmax=1),
                 alpha=0.5)

    
    ucb = upper_confidence_bound(state_space, gp_model)
    idx_new = np.argmax(ucb)
    x_new = state_space[idx_new,0]
    y_new = state_space[idx_new,1]
    sample_x = np.append(sample_x, x_new)
    sample_y = np.append(sample_y, y_new)
    cost_z = np.append(cost_z, black_box_cost(x_new, y_new))
    print(x_new, y_new)
    plt.savefig("./img/{}.png".format(i))