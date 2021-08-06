

import csv
import numpy as np  # http://www.numpy.org
import ast
from datetime import datetime
from math import log, floor, ceil
import random
import numpy as np

class Utility(object):

    # This method computes entropy for information gain
    def entropy(self, class_y):
        # Input:
        #   class_y         : list of class labels (0's and 1's)

        # TODO: Compute the entropy for a list of classes
        #
        # Example:
        #    entropy([0,0,0,1,1,1,1,1,1]) = 0.918 (rounded to three decimal places)

        entropy = 0

        #-pAlogpA - pBlogpB
        Nb = 0
        Na = 0
        for element in class_y:
            if element == 0:
                Na = Na + 1
            else:
                Nb = Nb + 1
        pa = Na/(Na+Nb)
        pb= Nb/(Na+Nb)



        if pa==0 or pb==0:
            return entropy

        entropy = -1*pa*log(pa, 2) -pb*log(pb, 2)



        #############################################
        return entropy


    def partition_classes(self, X, y, split_attribute, split_val):
        # Inputs:
        #   X               : data containing all attributes
        #   y               : labels
        #   split_attribute : column index of the attribute to split on
        #   split_val       : a numerical value to divide the split_attribute



        # : Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
        #
        # Split_val should be a numerical value
        # For example, the split_val could be the mean of the values of split_attribute
        #
        # Numeric Split Attribute:
        #   Split the data X into two lists(X_left and X_right) where the first list has all
        #   the rows where the split attribute is less than or equal to the split value, and the
        #   second list has all the rows where the split attribute is greater than the split
        #   value. Also create two lists(y_left and y_right) with the corresponding y labels.




        '''
        Example:



        X = [[3, 10],                 y = [1,
             [1, 22],                      1,
             [2, 28],                      0,
             [5, 32],                      0,
             [4, 32]]                      1]



        Here, columns 0 and 1 represent numeric attributes.



        Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
        Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.



        X_left = [[3, 10],                 y_left = [1,
                  [1, 22],                           1,
                  [2, 28]]                           0]



        X_right = [[5, 32],                y_right = [0,
                   [4, 32]]                           1]


 ([[3, 10], [1, 22], [2, 28]], [[5, 32], [4, 32]], [1, 1, 0], [0, 1])
        '''

        X_left = []
        X_right = []

        y_left = []
        y_right = []


        for i, row in enumerate(X):
            if row[split_attribute] > split_val:
                X_right.append(row)
                y_right.append(y[i])
            if row[split_attribute] <= split_val:
                X_left.append(row)
                y_left.append(y[i])

        #############################################
        return (X_left, X_right, y_left, y_right)


    def information_gain(self, previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value

        # : Compute and return the information gain from partitioning the previous_y labels
        # into the current_y labels.
        # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

        """
        Example:

        previous_y = [0,0,0,1,1,1]
        current_y = [[0,0], [1,1,1,0]]

        info_gain = 0.45915
        """
        info_gain = 0

        avg_entropy = 0
#         for node in current_y:
        avg_entropy = (self.entropy(current_y[0])*len(current_y[0]))/len(previous_y)+(self.entropy(current_y[1])*len(current_y[1]))/len(previous_y)

        info_gain = self.entropy(previous_y) - avg_entropy

        return info_gain


    def best_split(self, X, y):
        # Inputs:
        #   X       : Data containing all attributes
        #   y       : labels
        #              : For each node find the best split criteria and return the split attribute,
        #             spliting value along with  X_left, X_right, y_left, y_right (using partition_classes)
        #             in the dictionary format {'split_attribute':split_attribute, 'split_val':split_val,
        #             'X_left':X_left, 'X_right':X_right, 'y_left':y_left, 'y_right':y_right, 'info_gain':info_gain}
        '''

        Example:

        X = [[3, 10],                 y = [1,
             [1, 22],                      1,
             [2, 28],                      0,
             [5, 32],                      0,
             [4, 32]]                      1]

        Starting entropy: 0.971

        Calculate information gain at splits: (In this example, we are testing all values in an
        attribute as a potential split value, but you can experiment with different values in your implementation)

        feature 0:  -->    split_val = 1  -->  info_gain = 0.17
                           split_val = 2  -->  info_gain = 0.01997
                           split_val = 3  -->  info_gain = 0.01997
                           split_val = 4  -->  info_gain = 0.32
                           split_val = 5  -->  info_gain = 0

                           best info_gain = 0.32, best split_val = 4


        feature 1:  -->    split_val = 10  -->  info_gain = 0.17
                           split_val = 22  -->  info_gain = 0.41997
                           split_val = 28  -->  info_gain = 0.01997
                           split_val = 32  -->  info_gain = 0

                           best info_gain = 0.4199, best split_val = 22


       best_split_feature: 1
       best_split_val: 22



       'X_left': [[3, 10], [1, 22]]
       'X_right': [[2, 28],[5, 32], [4, 32]]

       'y_left': [1, 1]
       'y_right': [0, 0, 1]
        '''

        split_attribute = 0
        split_val = 0
        X_left, X_right, y_left, y_right = [], [], [], []

        output_dict = {}
        max_IG = 0

        best_split_feature, best_split_val, best_X_left, best_X_right, best_y_left,best_y_right = 0,0,[],[],[],[]

        features = range(len(X[0]))
        numberOfsamples = ceil(len(X[0])**(.5))

        samples = random.sample(features,numberOfsamples)



        for i in samples:

            split_attribute = i

            unique_x = np.unique([item[i] for item in X])


            for val in unique_x:
                split_val = val

                (X_left, X_right, y_left, y_right) = self.partition_classes( X, y, split_attribute, split_val)


                if ((len(y_left)==0) or (len(y_right)==0)):

                    continue


                previous_y = y
                current_y = [y_right, y_left]
                IG = self.information_gain(previous_y, current_y)
                if IG >= max_IG:

                    max_IG = IG
                    best_split_feature = split_attribute
                    best_split_val = split_val
                    best_X_left = X_left
                    best_X_right = X_right
                    best_y_left = y_left
                    best_y_right = y_right


        output_dict['best_split_feature'] = best_split_feature
        output_dict['best_split_val'] = best_split_val
        output_dict['info_gain'] = max_IG
        output_dict['X_left'] = best_X_left
        output_dict['X_right'] = best_X_right
        output_dict['y_left'] = best_y_left
        output_dict['y_right'] = best_y_right


        return output_dict

class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth

    	
    def learn(self, X, y, par_node = {}, depth=0):


        # : Train the decision tree (self.tree) using the the sample X and labels y

        # par_node is a parameter that is useful to pass additional information to call
        # the learn method recursively. Its not mandatory to use this parameter

        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)

        if depth >= self.max_depth:
            majority = max(set(y), key = y.count)
            return majority

        elif len(y) <= 1:
            majority = max(set(y), key = y.count)
            return majority


        elif all(x == y[0] for x in y):
            majority = max(set(y), key = y.count)
            return majority

        elif Utility().entropy(y) < 0.002:
            majority = max(set(y), key = y.count)
            return majority


        else:
            output_dict = Utility().best_split(X, y)


            par_node={}
            par_node['best_split_feature'] = output_dict['best_split_feature']
            par_node['best_split_val'] = output_dict['best_split_val']

            par_node['right'] = self.learn(output_dict['X_right'], output_dict['y_right'], {}, depth+1)


            par_node['left'] = self.learn(output_dict['X_left'], output_dict['y_left'], {}, depth+1)

            self.tree = par_node

            return par_node




    def classify(self, record):
        # : classify the record using self.tree and return the predicted label

        mytree = self.tree
        while isinstance(mytree, dict):

            if record[mytree['best_split_feature']] <= mytree['best_split_val']:
                mytree = mytree['left']

            else:
                mytree = mytree['right']

        return mytree





"""
Here,
1. X is assumed to be a matrix with n rows and d columns where n is the
number of total records and d is the number of features of each record.
2. y is assumed to be a vector of labels of length n.
3. XX is similar to X, except that XX also contains the data label for each
record.
"""

class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree(max_depth=10) for i in range(num_trees)]
        self.bootstraps_datasets = []
        self.bootstraps_labels = []

    def _bootstrapping(self, XX, n):
        # Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        #
        # : Create a sample dataset of size n by sampling with replacement
        #       from the original dataset XX.

        sample = [] # sampled dataset
        labels = []  # class labels for the sampled records

        labels = np.random.choice(list(range(len(XX))), n, replace=True)
        for i in labels:
            sample.append(XX[i])
        return (sample, labels)

    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        # : Train `num_trees` decision trees using the bootstraps datasets
        # and labels by calling the learn function from your DecisionTree class.

        X = []
        y = []
        for i,sample in enumerate(self.bootstraps_datasets):
            for record in sample:
                X.append(record[0:(len(record)-1)])
                y.append(record[-1])


            self.decision_trees[i].learn(X, y)


        pass
        #############################################

    def voting(self, X):
        y = []


        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []

            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]

                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)

                    votes.append(effective_vote)

            counts = np.bincount(votes)
            if len(counts) == 0:

                index = self.bootstraps_datasets[0].index(record)
                y = np.append(y, self.bootstraps_labels[0][index])

            else:
                y = np.append(y, np.argmax(counts))

        return y

    def user(self):
        """
        :return: string
         GTUsername
        """
        return 'samoozegar3'



def get_forest_size():
    forest_size = 10
    return forest_size

# : Determine random seed to set for reproducibility
def get_random_seed():
    random_seed = 0
    return random_seed