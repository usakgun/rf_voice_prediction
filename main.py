import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'voice.csv')
data = pd.read_csv(file_path)

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.drop('label', axis=1)
y = data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("Linear Discriminant Analysis Results")
print("-" * 30)

log_model = LogisticRegression()
log_model.fit(X_train_lda, y_train)
y_pred_log = log_model.predict(X_test_lda)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
ConfusionMatrixDisplay.from_estimator(log_model, X_test_lda, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_lda, y_train)
y_pred_tree = tree_model.predict(X_test_lda)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
ConfusionMatrixDisplay.from_estimator(tree_model, X_test_lda, y_test)
plt.title("Decision Tree Confusion Matrix")
plt.show()

print("\nSupport Vector Machines Results")
print("-" * 30)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_lda, y_train)
y_pred_linear = svm_linear.predict(X_test_lda)
print("SVM Linear Accuracy:", accuracy_score(y_test, y_pred_linear))
print(classification_report(y_test, y_pred_linear))
ConfusionMatrixDisplay.from_estimator(svm_linear, X_test_lda, y_test)
plt.title("SVM Linear Confusion Matrix")
plt.show()

svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train_lda, y_train)
y_pred_poly = svm_poly.predict(X_test_lda)
print("SVM Polynomial Accuracy:", accuracy_score(y_test, y_pred_poly))
print(classification_report(y_test, y_pred_poly))
ConfusionMatrixDisplay.from_estimator(svm_poly, X_test_lda, y_test)
plt.title("SVM Polynomial Confusion Matrix")
plt.show()

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_lda, y_train)
y_pred_rbf = svm_rbf.predict(X_test_lda)
print("SVM RBF Accuracy:", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))
ConfusionMatrixDisplay.from_estimator(svm_rbf, X_test_lda, y_test)
plt.title("SVM RBF Confusion Matrix")
plt.show()