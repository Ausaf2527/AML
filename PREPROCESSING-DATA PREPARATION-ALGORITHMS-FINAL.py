#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import imblearn
from imblearn.over_sampling import SMOTE


# In[2]:


data = pd.read_csv('PS_20174392719_1491204439457_log.csv')


# In[3]:


data


# In[4]:


df = data.sample(n=63626, replace=True)
df.reset_index(drop=True, inplace=True)


# In[5]:


df


# In[6]:


rate_sampled = df['isFraud'].value_counts()
rate_sampled


# In[7]:


ratio_sampled = rate_sampled/len(df.index)
print(f'The Ratio of fraudulent cases in the sampled data is:{ratio_sampled[1]}\nThe Ratio of non-fraudulent in the sampled data is:{ratio_sampled[0]}')


# In[8]:


df['step_day_hour'] = (df['step']) % 24
df


# In[9]:


df['step_day_week'] = (df['step']) % 7
df


# In[10]:


df['step_day_month'] = (df['step']) // 24
df


# In[11]:


df['nameOrig'] = df['nameOrig'].str[0]
df.head()


# In[12]:


df['nameDest'] = df['nameDest'].str[0]
df.head()


# In[13]:


df = pd.get_dummies(data = df, columns = ['type'] )
df.head()


# In[14]:


df = pd.get_dummies(data = df, columns = ['nameOrig'])
df = pd.get_dummies(data = df, columns = ['nameDest'])
df.head()


# In[15]:


## binning step into different time 

step_bin = [-1, 5, 11, 18, 24]
label_bin = ['midnight', 'morning', 'afternoon', 'evening']

df['time'] = pd.cut(df['step_day_hour'], bins = step_bin, labels = label_bin )
df.head()


# In[16]:


df = pd.get_dummies(data=df, columns=['time'])
df


# In[17]:


df = df.drop(columns = ['nameOrig_C', 'isFlaggedFraud'])
df.head()


# In[18]:


plt.figure(figsize=(22,25))
sns.heatmap(df.corr(),  annot = True)


# In[19]:


correlation = df.corr()
correlation['isFraud'].sort_values(ascending = False)


# In[20]:


X = df.drop(columns = ['isFraud'])
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 42, test_size = 0.2)


# In[21]:


sm = SMOTE(random_state = 42)


# In[22]:


X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)


# In[23]:


df_smote = pd.concat([X_train_sm, y_train_sm], axis = 1)
df_smote['isFraud'].value_counts()


# In[24]:


df_smote


# In[25]:


X_train_sm.head()


# In[26]:


#DECISION TREE
DT_base =DecisionTreeClassifier()
DT_base.fit(X_train_sm, y_train_sm)
y_pred_DT_base = DT_base.predict(X_test)
y_pred_DT_base_train = DT_base.predict(X_train_sm)


# In[27]:


recall_DT_base = recall_score(y_test, y_pred_DT_base)
acc_DT_base = accuracy_score(y_test, y_pred_DT_base)
acc_DT_base_train = accuracy_score(y_train_sm, y_pred_DT_base_train)
precision_DT_base = precision_score(y_test, y_pred_DT_base)
f1_DT_base = f1_score(y_test, y_pred_DT_base)
recall_DT_base_train = recall_score(y_train_sm, y_pred_DT_base_train)

print(f"train recall: {recall_DT_base_train}")
print(f"test recall: {recall_DT_base}")


# In[28]:


print(classification_report(y_test, y_pred_DT_base))


# In[29]:


cm_DT_base = confusion_matrix(y_test, y_pred_DT_base, labels = [1,0])
df_DT_base = pd.DataFrame(data = cm_DT_base, index = ['actual 1', 'actual 0'], columns = ['Predicted 1', 'Predicted 0'])
df_DT_base


# In[30]:


sns.heatmap(df_DT_base, annot = True)


# In[31]:


tree.plot_tree(DT_base)


# In[32]:


#LOGISTIC REGRESSION
logreg =LogisticRegression()
logreg.fit(X_train_sm, y_train_sm)
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_train = logreg.predict(X_train_sm)


# In[33]:


recall_logreg = recall_score(y_test, y_pred_logreg)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_logreg_train = accuracy_score(y_train_sm, y_pred_logreg_train)
precision_logreg = precision_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)
recall_logreg_train = recall_score(y_train_sm, y_pred_logreg_train)

print(f"train recall: {recall_logreg_train}")
print(f"test recall: {recall_logreg}")


# In[34]:


print(classification_report(y_test, y_pred_logreg))


# In[35]:


cm_logreg = confusion_matrix(y_test, y_pred_logreg, labels = [1,0])
df_logreg = pd.DataFrame(data = cm_logreg, index = ['actual 1', 'actual 0'], columns = ['Predicted 1', 'Predicted 0'])
df_logreg


# In[36]:


sns.heatmap(df_logreg, annot = True)


# In[37]:


#GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
gnb.fit(X_train_sm, y_train_sm)
y_pred_gnb = gnb.predict(X_test)
y_pred_gnb_train = gnb.predict(X_train_sm)


# In[38]:


recall_gnb = recall_score(y_test, y_pred_gnb)
acc_gnb = accuracy_score(y_test, y_pred_gnb)
acc_gnb_train = accuracy_score(y_train_sm, y_pred_gnb_train)
precision_gnb = precision_score(y_test, y_pred_gnb)
f1_gnb = f1_score(y_test, y_pred_gnb)
recall_gnb_train = recall_score(y_train_sm, y_pred_gnb_train)

print(f"train recall: {recall_gnb_train}")
print(f"test recall: {recall_gnb}")


# In[39]:


print(classification_report(y_test, y_pred_gnb))


# In[40]:


cm_gnb = confusion_matrix(y_test, y_pred_gnb, labels = [1,0])
df_gnb = pd.DataFrame(data = cm_gnb, index = ['actual 1', 'actual 0'], columns = ['Predicted 1', 'Predicted 0'])
df_gnb


# In[41]:


sns.heatmap(df_gnb, annot =True)


# In[42]:


#K-NEAREST NEIBHOURS
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
knn.fit(X_train_sm, y_train_sm)


# In[43]:


y_pred_knn = knn.predict(X_test)
y_pred_knn_train = knn.predict(X_train_sm)


# In[44]:


recall_knn = recall_score(y_test, y_pred_knn)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_knn_train = accuracy_score(y_train_sm, y_pred_knn_train)
precision_knn = precision_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
recall_knn_train = recall_score(y_train_sm, y_pred_knn_train)

print(f"train recall: {recall_knn_train}")
print(f"test recall: {recall_knn}")


# In[45]:


print(classification_report(y_test, y_pred_knn))


# In[46]:


cm_knn = confusion_matrix(y_test, y_pred_knn, labels = [1,0])
df_knn = pd.DataFrame(data = cm_knn, index = ['actual 1', 'actual 0'], columns = ['Predicted 1', 'Predicted 0'])
df_knn


# In[47]:


sns.heatmap(df_knn, annot = True)


# In[48]:


#SUPPORT VECTOR MACHINE
svc_model = SVC()
svc_model.fit(X_train_sm, y_train_sm)


# In[49]:


y_pred_svc_model = svc_model.predict(X_test)
y_pred_svc_model_train = svc_model.predict(X_train_sm)


# In[50]:


recall_svc_model = recall_score(y_test, y_pred_svc_model)
acc_svc_model = accuracy_score(y_test, y_pred_svc_model)
acc_svc_model_train = accuracy_score(y_train_sm, y_pred_svc_model_train)
precision_svc_model = precision_score(y_test, y_pred_svc_model)
f1_svc_model = f1_score(y_test, y_pred_svc_model)
recall_svc_model_train = recall_score(y_train_sm, y_pred_svc_model_train)

print(f"train recall: {recall_svc_model_train}")
print(f"test recall: {recall_svc_model}")


# In[51]:


print(classification_report(y_test, y_pred_svc_model))


# In[52]:


cm_svc_model = confusion_matrix(y_test, y_pred_svc_model, labels = [1,0])
df_svc_model = pd.DataFrame(data = cm_svc_model, index = ['actual 1', 'actual 0'], columns = ['Predicted 1', 'Predicted 0'])
df_svc_model


# In[53]:


sns.heatmap(df_svc_model, annot = True)


# In[54]:


eva_met_comparison = {
    "LogisticReg": [acc_logreg,precision_logreg,recall_logreg,f1_logreg],
    "KNN": [acc_knn, precision_knn, recall_knn, f1_knn],
    "DecisionTree": [acc_DT_base, precision_DT_base, recall_DT_base, f1_DT_base] ,
    "Naive Bayes": [acc_gnb, precision_gnb, recall_gnb, f1_gnb] ,
    " Support Vector Machine-SVC": [acc_svc_model, precision_svc_model, recall_svc_model, f1_svc_model] ,
}

eva_comparison = pd.DataFrame(data = eva_met_comparison, index = ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
eva_comparison


# In[55]:


# Performance Comparison
algorithms = ['LogisticReg', 'KNN', 'DecisionTree','Naive Bayes','Support Vector Machine-SVC']
accuracy = [acc_logreg,acc_knn,acc_DT_base,acc_gnb,acc_svc_model]
precision = [precision_logreg,precision_knn,precision_DT_base,precision_gnb,precision_svc_model]
recall = [recall_logreg,recall_knn,recall_DT_base,recall_gnb,recall_svc_model]
f1 = [f1_logreg,f1_knn,f1_DT_base,f1_gnb,f1_svc_model]

bar_width = 0.2

index = np.arange(len(algorithms))

plt.figure(figsize=(10, 6))

plt.bar(index, accuracy, bar_width, color='blue', label='Accuracy')
plt.bar(index + bar_width, precision, bar_width, color='green', label='Precision')
plt.bar(index + 2 * bar_width, recall, bar_width, color='orange', label='Recall')
plt.bar(index + 3 * bar_width, f1, bar_width, color='purple', label='F1')

plt.xlabel('Algorithms')
plt.ylabel('Metrics')
plt.title('Comparison of Performance Metrics')
plt.xticks(index + bar_width, algorithms)
plt.legend()

plt.tight_layout()
plt.show()


# In[57]:


recall_comparison = {
    "LogisticReg": [recall_logreg_train, recall_logreg],
    "KNN": [recall_knn_train, recall_knn],
    "DecisionTree": [recall_DT_base_train, recall_DT_base],
    "Naive Bayes": [recall_gnb_train, recall_gnb],
    "Support Vector Machine-SVC": [recall_svc_model_train, recall_svc_model],
}

recall_df = pd.DataFrame(data=recall_comparison, index=['Recall (Training)', 'Recall (Test)'])
recall_df


# In[58]:


acc_comparison = {
    "LogisticReg": [acc_logreg_train, acc_logreg],
    "KNN": [acc_knn_train, acc_knn],
    "DecisionTree": [acc_DT_base_train, acc_DT_base],
    "Naive Bayes": [acc_gnb_train, acc_gnb],
    "Support Vector Machine-SVC": [acc_svc_model_train, acc_svc_model],
}

acc_df = pd.DataFrame(data=acc_comparison, index=['Accuracy (Training)', 'Accuracy (Test)'])
acc_df


# In[59]:


algorithms = list(acc_comparison.keys())
train_accuracies = [item[0] for item in acc_comparison.values()]
test_accuracies = [item[1] for item in acc_comparison.values()]

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(algorithms))

bar1 = ax.bar(index, train_accuracies, bar_width, label='Training Accuracy')
bar2 = ax.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy')

ax.set_xlabel('Algorithms')
ax.set_ylabel('Accuracy')
ax.set_title('Comparative Analysis of Accuracy on Training and Test Data')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(algorithms, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()


# In[60]:


from sklearn.metrics import roc_curve, auc

y_true = y_test

fpr_lr, tpr_lr, _ = roc_curve(y_true, y_pred_logreg)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_dt, tpr_dt, _ = roc_curve(y_true, y_pred_DT_base)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_gnb, tpr_gnb, _ = roc_curve(y_true, y_pred_gnb)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

fpr_svc, tpr_svc, _ = roc_curve(y_true, y_pred_svc_model)
roc_auc_svc = auc(fpr_svc, tpr_svc)

fpr_knn, tpr_knn, _ = roc_curve(y_true, y_pred_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curves for all models
plt.figure(figsize=(20, 10))
plt.plot(fpr_lr, tpr_lr, color='blue', label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, color='green', label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot(fpr_gnb, tpr_gnb, color='red', label=f'Random Forest (AUC = {roc_auc_gnb:.2f})')
plt.plot(fpr_svc, tpr_svc, color='purple', label=f'Support Vector Machine (AUC = {roc_auc_svc:.2f})')
plt.plot(fpr_knn, tpr_knn, color='orange', label=f'K-Nearest Neighbors (AUC = {roc_auc_knn:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[61]:


from sklearn.metrics import precision_recall_curve

# Computing Precision-Recall curve and AUC for each model
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_logreg)
precision_dt, recall_dt, _ = precision_recall_curve(y_test, y_pred_DT_base)
precision_gnb, recall_gnb, _ = precision_recall_curve(y_test, y_pred_gnb)
precision_svc, recall_svc, _ = precision_recall_curve(y_test, y_pred_svc_model)
precision_knn, recall_knn, _ = precision_recall_curve(y_test, y_pred_knn)

# Plotting Precision-Recall curves for all models
plt.figure(figsize=(20, 10))
plt.plot(recall_lr, precision_lr, color='blue', label='Logistic Regression')
plt.plot(recall_dt, precision_dt, color='green', label='Decision Tree')
plt.plot(recall_gnb, precision_gnb, color='red', label='Gaussian Naive Bayes')
plt.plot(recall_svc, precision_svc, color='purple', label='Support Vector Machine')
plt.plot(recall_knn, precision_knn, color='orange', label='K-Nearest Neighbors')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()


# In[ ]:




