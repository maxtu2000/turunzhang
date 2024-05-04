import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve,roc_curve,auc,roc_auc_score
from sklearn.metrics import confusion_matrix
data=[]
with open("Get_vectors.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
# print(type(X[0]))
# print(type(X[0][0]))
# print(X[0][0])
Y=data[2]
y=[]
for k in Y:
    y.append(int(k))

x=[]
for i in data[0]:
    temp=[]
    i=i.lstrip('[')
    i=i.rstrip(']')
    i=i.split(',')
    for j in i:
        j=j.strip("'")
        j=float(j)
        temp.append(j)
    x.append(temp)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 0)
scores = cross_val_score(clf, x, y, cv=5)
print(scores)
print("准确率:",scores.mean())
print("方差:", scores.var())
print("标准差:", scores.std())



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 0)
classifier.fit(X_train, y_train)
score=classifier.score(X_test,y_test)
# print(score)
y_pred=classifier.predict(X_test)
f1=f1_score(y_test, y_pred, average='binary')
precision=precision_score(y_test, y_pred, average='binary')
recall=recall_score(y_test, y_pred, average='binary')
print('f1:',f1)
print('precision:',precision)
print('recall:',recall)

con_matrix=confusion_matrix(y_test, y_pred, labels=None)
print(con_matrix)

fig,ax = plt.subplots(figsize=(12,10))
lr_roc = plot_roc_curve(estimator=classifier, X=X_test,
                        y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.show()

probas_ = classifier.predict_proba(X_test)
precision,recall,thresholds = precision_recall_curve(y_test,probas_[:,1])# 最重要的函数：通过precision_recall_curve()函数，求出recall，precision，以及阈值
plt.plot(recall,precision,lw=1)
plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck")				 # 画对角线
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel("Recall Rate")
plt.ylabel("Precision Rate")
plt.show()
