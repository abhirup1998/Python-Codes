import matplotlib.pyplot as plt
import random
import scipy.stats as ss
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
votes = [1,2,3,1,2,3,1,2,3,3,3,3,2,2,2,1,1]
vote_counts = majority_vote(votes)

##returns the vote# that is most common
#%%
    winners_eqiv = ss.mstats.mode(votes)