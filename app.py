import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from flask import Flask
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)


def classifyDriver(driverID):
    fileName = driverID + ".csv"
    data = pd.read_csv(fileName, header=1)
    data.columns = ['speed', 'distance','timestamp','label']

    def dayOrNight(timestamp):
        time = 0    #time = 0 means it's day
        hour = timestamp.split(" ")[1].split(":")[0]
        if(int(hour) < 5 or int(hour) > 19):
            time = 1    # time = 1 means it's night
        return time

    backupData = data.copy(deep=True)
    data["timestamp"] = data["timestamp"].apply(dayOrNight)

    factor = pd.factorize(data['label'])

    print(type(factor))
    data.label = factor[0]
    definitions = factor[1]

    #Splitting the data into independent and dependent variables
    X = data.iloc[:,0:3].values
    y = data.iloc[:,3].values
    print('The independent features set: ')
    print(X[:5,:])
    print('The dependent variable: ')
    print(y[:5])

    # Creating the Training and Test set from data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    print(type(X_test))
    print(classifier.predict([180,100,1]))
#     #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
#     reversefactor = dict(zip(range(3),definitions))
#     y_test = np.vectorize(reversefactor.get)(y_test)
#     y_pred = np.vectorize(reversefactor.get)(y_pred)
#     # Making the Confusion Matrix
#     print(pd.crosstab(y_test, y_pred, rownames=['Actual driver category'], colnames=['Predicted driver category']))

#     print(list(zip(data.columns[0:4], classifier.feature_importances_)))
#     # joblib.dump(classifier, 'randomforestmodel.pkl') 
#     return "safe"

# class Driver(Resource):
#     def get(self, driverID):
#         return classifyDriver(driverID)

# api.add_resource(Driver, "/driver/<string:driverID>")
# app.run(debug=True)