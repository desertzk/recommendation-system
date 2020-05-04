'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat


    def load_train_test_data(self):
        rating_data = pd.read_csv('Data/ratings.dat', sep="::")
        rating_data.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
        groupby_data = rating_data.groupby(['MovieID'])
        idx = rating_data.groupby(['UserID'])['Timestamp'].idxmax()
        sort_res = groupby_data.apply(lambda x: x.sort_values(by='Timestamp', ascending=False))

        test_list = []
        for index in idx:
            first_test = []
            UserID = rating_data.get_value(index, 'UserID')
            MovieID = rating_data.get_value(index, 'MovieID')
            first_test.append(UserID)
            first_test.append(MovieID)
            test_list.append(first_test)
            rating_data.drop(index=index)

        # Construct matrix
        mat = sp.dok_matrix((6041, 3953), dtype=np.float32)
        for e in rating_data.values:
            if e[2]>0:
                mat[e[0] - 1, e[1] - 1] = 1.0
            # Mat[e[0] - 1][e[1] - 1] = e[2]

        # print(test_list)
        return mat,test_list


if __name__ == '__main__':
    dataset = Dataset('Data/ml-1m')
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done . #user=%d, #item=%d, #train=%d, #test=%d"
           ,num_users, num_items, train.nnz, len(testRatings))


    rating_data = pd.read_csv('Data/ratings.dat', sep="::")
    rating_data.columns = ["UserID", "MovieID", "Rating", "Timestamp"]
    groupby_data = rating_data.groupby(['MovieID'])
    idx= rating_data.groupby(['UserID'])['Timestamp'].idxmax()
    sort_res = groupby_data.apply(lambda x: x.sort_values(by='Timestamp',ascending=False))

    test_list=[]
    for index in idx:
        first_test = []
        UserID = rating_data.get_value(index,'UserID')
        MovieID = rating_data.get_value(index,'MovieID')
        first_test.append(UserID)
        first_test.append(MovieID)
        test_list.append(first_test)
        rating_data.drop(index=index)

    # Construct matrix
    mat = sp.dok_matrix((6040, 3952), dtype=np.float32)
    for e in rating_data.values:
        if e[2]>0:
            mat[e[0] - 1, e[1] - 1] = 1.0
        # Mat[e[0] - 1][e[1] - 1] = e[2]

    print(test_list)

