import pandas as pd
import numpy as np
import pickle
from sklearn import svm

from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc


df = pd.read_csv(
    r"D:\dementia\DementiaPrediction\DementiaPrediction\oasis_varied.csv")


df.drop(columns=['ID'], inplace=True)


# Handing missing data
df.fillna(0, inplace=True)

# to transform categorical data into numeric format
label_encoder = LabelEncoder()
df['M/F'] = label_encoder.fit_transform(df['M/F'])
# df['Group'] = label_encoder.fit_transform(df['Group'])

df['Hand'] = label_encoder.fit_transform(df['Hand'])
# print(df.to_string())
X = df.drop(columns=['MMSE'], axis=1)
y = df['MMSE']

print(df.dtypes)
y_binary = (y > np.median(y)).astype(int)

# # Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=None)


# Standardize features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)
print(y_pred)
accuracy_clf = accuracy_score(y_test, y_pred)
print("Accuracy of Random Forest: {:.2f}%".format(accuracy_clf * 100))
f_accuracy_clf = accuracy_clf*100

# Pickle the object
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
