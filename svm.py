import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix : ")
print(confusion_matrix)

classification_report = metrics.classification_report(y_test, y_pred)
print("Classification Report : ")
print(classification_report)

plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r)
    plt.title(f"Predicted : {y_pred[i]}, Actual : {y_test[i]}")
    plt.axis('on')