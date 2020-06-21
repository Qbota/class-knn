from math import pow
from math import sqrt


class KNeighboursClassifier:

    __X_train = []
    __y_train = []
    __neighbours = int

    def fit(self, X_train, y_train, neighbours=5):
        if len(X_train) != len(y_train):
            raise Exception('Size of both data lists should be equal')
        if neighbours % 2 == 0:
            raise Exception('Number of neighbours must be odd')
        self.__X_train = X_train
        self.__y_train = y_train
        self.__neighbours = neighbours


    def predict(self, X_test):
        listOfDistances = []
        predictions = []
        for test in range(len(X_test)):
            distances = []
            for train in range(len(self.__X_train)):
                distances.append((train, self.calculateDistance(X_test[test], self.__X_train[train])))
            listOfDistances.append(distances)
        for distances in listOfDistances:
            distances.sort(key=lambda tup: tup[1])
            labels = []
            for distance in distances[:self.__neighbours]:
                labels.append(self.__y_train[distance[0]])
            predictions.append(int(sum(labels) / len(labels)))
        return predictions

    def calculateDistance(self, first, second):
        if len(first) != len(second):
            raise Exception('Size should be equal')
        squareOfDistance = 0
        for i in range(len(first)):
            squareOfDistance += pow(first[i] - second[i], 2)
        distance = sqrt(squareOfDistance)
        return distance