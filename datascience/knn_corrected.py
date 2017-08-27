from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
#%%
def generate_synth_data(n=50):
    sample = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(2,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (sample, outcomes)
#%%
def plot_prediction_grid (xx, yy, p_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, p_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)
#%%
def make_prediction_grid(predictors, limits, h, k=50):
    # y values are the rows and x values are the columns
    (x_min,x_max,y_min,y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    k_means = KMeans(n_clusters=2, random_state=0).fit(predictors)
    
    prediction_grid = np.zeros(xx.shape, dtype = int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = k_means.predict([p])
    return (xx, yy, prediction_grid)
#%%
(predictors, outcomes) = generate_synth_data(50)
#%%
k = 40; filename = "knn_synth_50.pdf"; limits = (-3, 4, -3, 4); h = 0.1
(xx, yy, prediction_grid) = make_prediction_grid(predictors, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)
