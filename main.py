from KNeighboursClassifier import KNeighboursClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from numpy import genfromtxt

def main():
    data = genfromtxt('diabetes_csv.csv', delimiter=',')
    X = data[:,:0-1]
    y = data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = KNeighboursClassifier()
    clf.fit(X_train, y_train, 5)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    main()
