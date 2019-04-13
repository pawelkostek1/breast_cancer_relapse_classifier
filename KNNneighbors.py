# Assigning features and label variables
#https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

fr = open("feature_matrix", "r")

values = []
for i, line in enumerate(fr):
    if(i != 0):
        lineList = (line.strip().split('\t'))
        print(lineList[1:])
        values.append(lineList[1:])

fr.close()
theLabels = values[-1] #class labels
values = values[0:len(values) -1] #remove class labels


arrayOfPoints = []

for i in range(len(values[0])):
    aPoint = []
    for item in values:
        aPoint.append(item[i])
    arrayOfPoints.append(tuple(aPoint))


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Predict Output
# predicted= model.predict([removeForPrediction]) # 0:Overcast, 2:Mild
# print(predicted)

from sklearn.model_selection import cross_val_score
import numpy as np
import operator

RANGE_K = 50
ITERS = 10
kVals = {}

for i in range(1, RANGE_K):
    avgs = []
    for j in range(ITERS):
        X_train, X_test, y_train, y_test = train_test_split(arrayOfPoints, theLabels, test_size=0.3, stratify=theLabels)
        #create a new KNN model
        knn_cv = KNeighborsClassifier(n_neighbors=i)
        #train model with cv of 6
        cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=6)
        #print each cv score (accuracy) and average them
        print(cv_scores)
        print("cv_scores mean:{}:".format(np.mean(cv_scores)))
        avgs.append(np.mean(cv_scores))
    print(i, avgs)
    kVals[i] = np.mean(avgs)
print(max(kVals.items(), key=operator.itemgetter(1)))
print(kVals) # RANGE_K values and average model accuracy for ITERS


#import seaborn as sns; sns.set()
#import matplotlib.pyplot as plt