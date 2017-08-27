import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

points = np.array([[5,1], [5,2], [6,1], [6,7], [3,6], [3,5], [7,6], [8,6], [2,5], [1,5]])
outcomes = np.array([0,0,0,0,0,1,1,1,1,1])

distances = np.zeros(np.shape(points))
for i in range(len(distances)):
    distances[i] = np.linalg.norm(p-points[i])

#loop over all the points
#    compute the distances between p and all other points
#sort those distances and return k points closest to p
#%%
def scatter2D(points):
    import matplotlib.pyplot as plt
    x = []
    y = []
    for p in points:
        x.append(p[0])
        y.append(p[1])
    plt.scatter(x, y)
    plt.show()
#%%
plt.plot(points[:, 0], points[:, 1], "ro")
plt.plot(p[0], p[1], "bo")
plt.show()
#%%
def majority_vote(votes):
    """
    """
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_count = max(vote_counts.values())
    for vote in vote_counts:
        if vote_counts[vote] == max_count:
            winners.append(vote)
    return random.choice(winners)
#%%
def find_k_nearest(points, p, k=4):
    distances = np.zeros(len(points))
    for i in range(0,len(distances)):
        distances[i] = np.linalg.norm(p - points[i])
    ind = np.argsort(distances)
    return ind
#%%
def knn_predict(p, points, outcomes, k=5):
    #find k nearest neighbours
    ind = find_k_nearest(points, p, k)
    #predict the class of p accordingly
    return majority_vote(outcomes[ind])
#%%
def make_prediction_grid(p, predictors, outcomes, limits, h, k):
    (x_min,x_max,y_min,y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, yy.shape, dtype= int)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)
    return (xx, yy, prediction_grid)
#%%
##GENERATING BIVARIATE COORDINATES
def generate_synth_data(n=50):
    sample = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (sample, outcomes)
#%%

n=20
plt.figure()
plt.plot(points[:n, 0], points[:n, 1], "ro")
plt.plot(points[n:, 0], points[n:, 1], "bo")
plt.savefig("bivardata.pdf")
