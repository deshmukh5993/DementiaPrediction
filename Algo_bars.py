import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(
    r"D:\dementia\DementiaPrediction\DementiaPrediction\oasis_cross-sectional.csv")


df.drop(columns=['ID', 'Delay'], inplace=True)


df.fillna(0, inplace=True)

# to transform categorical data into numeric format
label_encoder = LabelEncoder()
df['M/F'] = label_encoder.fit_transform(df['M/F'])
df['Hand'] = label_encoder.fit_transform(df['Hand'])

X = df.drop('MMSE', axis=1)
y = df['MMSE']

y_binary = (y > np.median(y)).astype(int)

# Split the data into training and testing setso
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_regression = LogisticRegression()
log_regression.fit(X_train, y_train)
y_pred_logistic = log_regression.predict(X_test)

accuracy_log = accuracy_score(y_test, y_pred_logistic)
print("Accuracy of Logistic Regression : {:.2f}%".format(accuracy_log * 100))
f_accuracy_log = accuracy_log*100

# AdaBoost
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
abc.fit(X_train, y_train)
y_pred_adaboost = abc.predict(X_test)

accuracy_abc = accuracy_score(y_test, y_pred_adaboost)
print("Accuracy of AdaBoost: {:.2f}%".format(accuracy_abc * 100))
f_accuracy_abc = accuracy_abc*100

# Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_desicion = dtree.predict(X_test)

accuracy_dtree = accuracy_score(y_test, y_pred_desicion)
print("Accuracy of Decision Tree: {:.2f}%".format(accuracy_dtree * 100))
f_accuracy_dtree = accuracy_dtree*100

# Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random = clf.predict(X_test)

accuracy_clf = accuracy_score(y_test, y_pred_random)
print("Accuracy of Random Forest: {:.2f}%".format(accuracy_clf * 100))
f_accuracy_clf = accuracy_clf*100

# SVM
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM: {:.2f}%".format(accuracy_svm * 100))
f_accuracy_svm = accuracy_svm*100

# KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of KNN: {:.2f}%".format(accuracy_knn * 100))
f_accuracy_knn = accuracy_knn*100

algorithms = ['Logistic Regression \n{:.2f}%'.format(accuracy_log * 100), 'AdaBoost\n{:.2f}%'.format(accuracy_abc * 100),
              'Decision Tree\n{:.2f}%'.format(
                  accuracy_dtree * 100), 'Random Forest\n{:.2f}%'.format(accuracy_clf * 100),
              'SVM\n{:.2f}%'.format(accuracy_svm * 100), 'KNN\n{:.2f}%'.format(accuracy_knn * 100)]
accuracy_values = [f_accuracy_log, f_accuracy_abc,
                   f_accuracy_dtree, f_accuracy_clf, f_accuracy_svm, f_accuracy_knn]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(algorithms, accuracy_values, color=[
        'blue', 'green', 'red', 'purple', 'black', 'brown'])


# Adding labels and title
# plt.yticks(np.arange(10, 110, 10))

plt.xlabel('Machine Learning Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy of Machine Learning Algorithms')

plt.show()
