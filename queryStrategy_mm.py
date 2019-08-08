import numpy as np
import random
import math
from similarities_mm import jaccard_similarity, cosine_similarity

class QueryStrategy(object):

    """Pool-based query strategy

    A QueryStrategy advices on which unlabeled data to be queried next given
    a pool of labeled and unlabeled data. Strategy = ['lc','sm','entropy','random']
    """


    def __init__(self, strategy, data, labels, pool, posterior_probs, budget, **kwargs):
        self._strategy = strategy
        self._data = data
        self._labels = labels
        self._posterior_probs = posterior_probs
        self._budget = budget

        if(np.shape(pool)[0] == 0 ):
            self._pool = self._data
        else:
            self._pool = pool

    @property
    def dataset(self):
        """The Dataset object that is associated with this QueryStrategy."""
        return self._data, self._labels

    def get_indices(self):
        return self.indices_batch

    def make_query(self):
        """Return the index of the sample to be queried and labeled. Read-only.

        No modification to the internal states.

        Returns
        -------
        sampled_data and sampled_labels

        """
        print('Strategy:', self._strategy)
        # already done, uncertainty sampling
        if self._strategy == 'lc':  # least confident
            score = -np.max(self._posterior_probs, axis=1)
            indices = np.argsort(score)[::-1]
            self.indices_batch = indices[:self._budget]

        # already done, random sampling
        elif self._strategy == 'random':
            self.indices_batch = random.sample(range(0, np.shape(self._data)[0]), self._budget)


        elif self._strategy == 'sm':  # smallest margin
            if np.shape(self._posterior_probs)[1] > 2:
                # Find 2 largest decision values
                self._posterior_probs = -(np.partition(self._posterior_probs, 2, axis=1)[:, :2])
            score = -np.abs(self._posterior_probs[:, 0] - self._posterior_probs[:, 1])
            indices = np.argsort(score)[::-1]
            self.indices_batch = indices[:self._budget]

        elif self._strategy == 'entropy':
            eps = 10**(-20) # adding eps in order to handle zero probability
            score = np.sum(-self._posterior_probs * np.log(self._posterior_probs + eps), axis=1)
            indices = np.argsort(score)[::-1]
            self.indices_batch = indices[:self._budget]
        
        elif self._strategy == 'diversity':
            # diversity = []
            # for example in self._data: #all data is unlabeled
            #     n = 0
            #     value = 0
            #     for point in self._pool:
            #         value += jaccard_similarity(example, point)
            #         n = n + 1
            #     value = value / n  # avarage value
            #     diversity.append(value)
            # indices = sorted(range(len(diversity)), key=lambda i: diversity[i])
            # self.indices_batch = indices[:self._budget]

            s = diveristySampling(self._data, pool = self._pool, budget = self._budget)
            s.updateCplus()
            # returns indices of # budget most diverse examples in data; if budget > size(data), returns only budget 
            self.indices_batch = s.newind


        sampled_data = self._data[self.indices_batch]
        sampled_labels = self._labels[self.indices_batch]
        # print('Sampled data shape',np.shape(sampled_data))
        # print('Sampled labels shape', np.shape(sampled_labels))

        return sampled_data, sampled_labels
