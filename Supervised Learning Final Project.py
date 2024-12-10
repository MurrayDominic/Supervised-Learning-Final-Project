#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This project focuses on predicting whether an individual’s annual income exceeds $50K based on census data, using a binary classification task. We will explore three types of supervised machine learning models: Logistic Regression, K-Nearest Neighbors, and Random Forest. 
# 
# By evaluating the accuracy and precision of each model, we aim to determine the best algorithm for modeling this data. 
# 
# Understanding income classification is important as it can help identify key factors influencing income levels.
# 

# # Data
# 
# The dataset used in this project is sourced from the UCI Machine Learning Repository and is an extraction from the 1994 Census database. The dataset can be accessed at https://archive.ics.uci.edu/dataset/2/adult.
# 
# The data is in tabular format, comprising 32,561 rows (samples) and 15 columns, with the target variable being "income." Out of these, 6 columns contain integer values, while the remaining columns are categorical and will be converted appropriately during preprocessing. Missing values are denoted by "?".
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, normaltest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[24]:


#import
df = pd.read_csv("adult.csv")

# remove blank spaces
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

categorical_features = df.select_dtypes(include=['object', 'category']).columns
numerical_features = df.select_dtypes(include=[np.number]).columns

print(f"Number of row: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"\nNumber of categorical features: {len(categorical_features)}")
print(f"Number of numerical features: {len(numerical_features)}")

print(f"\nInfo") 
df.info()


# # Data Cleaning
# 
# For the data cleaning process, we will begin by inspecting the unique values within each column. This will help us better understand the data.
# 
# We will also identify which columns contain the missing value marker, "?".
# 
# Finally we will look at the distribution of the data.

# In[3]:


# to see the unique values that are in each column
for c in df.columns[0:]:
    print(c, df[c].unique())


# In[4]:


#missing values
question_mark_count = (df == '?').sum()

print(question_mark_count)


# In[5]:


# distribution of numerical
for col in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


# In[6]:


# distribution of categorical 
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.show()


# In[7]:


# re-labelling
df.replace('?', np.nan, inplace=True)

# most frequent replacement
for column in df.columns:
    mode_value = df[column].mode()[0]  
    df[column].fillna(mode_value, inplace=True)  

# droping 'education' as there is already an ordered numberic version, 'education-num' in the data

df = df.drop(columns=['education'])

# adjusting 'native-country'

df['native-country'] = df['native-country'].apply(lambda x: 'Other' if x != 'United-States' else x)
df['race'] = df['race'].apply(lambda x: 'Other' if x != 'White' else x)


# The "education" column was droped, as it is redundant with the "education-num" column, which provides an ordered numeric version of the same information.
# 
# We identified that the columns "workclass," "occupation," and "native-country" contain missing values. We handled these missing values by replacing them with the most frequent value within each column.
# 
# Upon inspection of the distribution, the field "native-country" was dominated by the United States. Therefore, it was decided to group all other countries under "Other." The same was true "race" which was domiated by white. These adjustments should prevent collinearity problems later on.

# # Exploratory Data Analysis

# In[8]:


categorical_features = df.select_dtypes(include=['object', 'category']).columns
numerical_features = df.select_dtypes(include=[np.number]).columns

print("\nStatistical Summary (Numeric Features):")
print(df[numerical_features].describe())

print("\nStatistical Summary (Categorical Features):")
print(df[categorical_features].describe())


# In[9]:


# correlation matrix
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.show()


# In[10]:


# pair plot
sns.pairplot(df, diag_kind ='kde')


# Within the numerical columns, there does not appear to be any strong correlation between the fields.
# 
# From the statistical summary:
# 
# Both capital-gain and capital-loss are predominantly zeros.
#     
# None of the categorical features have a high number of unique values, with the highest count being 14.

# # Models
# 
# First we need to prepare the data to be modelled. This means converting catagorical fields into numerical.
# 
# The data is also split into training and test data.

# In[11]:


# adjusting cat cols into num 
df_adj = pd.get_dummies(df, columns=['workclass', 'marital-status', 'occupation', 'relationship', 'race', 
                                 'sex', 'native-country'], drop_first=True)

df_adj['income'] = df_adj['income'].str.strip()
df_adj['income'].replace('<=50K', '0', inplace = True)
df_adj['income'].replace('>50K', '1', inplace = True)
df_adj['income'] = df_adj['income'].astype(int)


# spliting the data
X = df_adj.drop(columns=['income']) 
y = df_adj['income']          
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)


# In[12]:


model_results = {
    "Model": ["LR", "KNN", "RF"],
    "Accuracy_Train": [0, 0, 0],
    "Accuracy_Test": [0, 0, 0],
    "Precision_Train": [0, 0, 0],
    "Precision_Test": [0, 0, 0]
}

model_results = pd.DataFrame(model_results)


# ## Collinearity
# 

# In[13]:


corr_matrix = X.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))


corr_values = corr_matrix.where(~mask).stack()

sorted_corr = corr_values.abs().sort_values(ascending=False)
top_corr = sorted_corr.head(10)

print(top_corr)


# As shown above, the highest correlation is 0.64, which is below the threshold of concern. Therefore, no further action is required on the data.

# # Logistic Regression

# ## Hyperparameter turning

# In[23]:


C_values = [0.001, 0.01, 0.1, 1, 10, 100] 


train_accuracy = {}
test_accuracy = {}
train_precision = {}
test_precision = {}

for C in C_values:
    train_accuracy[C] = []
    test_accuracy[C] = []
    train_precision[C] = []
    test_precision[C] = []
    
    lr = LogisticRegression(C=C) 
    lr.fit(X_train, y_train)
    train_pred = lr.predict(X_train)
    test_pred = lr.predict(X_test)

    # accuracy
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    train_accuracy[C].append(train_acc)
    test_accuracy[C].append(test_acc)

    # precision
    train_prec = precision_score(y_train, train_pred, average='weighted', zero_division=0)
    test_prec = precision_score(y_test, test_pred, average='weighted', zero_division=0)
    train_precision[C].append(train_prec)
    test_precision[C].append(test_prec)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(14, 10))


axs[0].plot(C_values, [train_accuracy[C][0] for C in C_values], label="Train Accuracy", marker="o", color="Green")
axs[0].plot(C_values, [test_accuracy[C][0] for C in C_values], label="Test Accuracy", marker="o", color="steelblue")
axs[0].set_title("Accuracy", fontsize=16)
axs[0].set_xlabel("C value", fontsize=12)
axs[0].set_ylabel("Accuracy", fontsize=12)
axs[0].set_xscale('log')  
axs[0].grid(alpha=0.3)
axs[0].legend(fontsize=12)


axs[1].plot(C_values, [train_precision[C][0] for C in C_values], label="Train Precision", marker="o", color="Green")
axs[1].plot(C_values, [test_precision[C][0] for C in C_values], label="Test Precision", marker="o", color="steelblue")
axs[1].set_title("Precision", fontsize=16)
axs[1].set_xlabel("C value", fontsize=12)
axs[1].set_ylabel("Precision", fontsize=12)
axs[1].set_xscale('log')  
axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=12)

plt.show()


# ## Final Logistic Regression Model
# 
# The results above indicate that adjusting the 'C' value has no effect on the accuracy or precision of the Logistic Regression model for this data. Therefore, we will proceed with the default values for the final comparison.
# 
# 

# In[15]:


lr = LogisticRegression() 
lr.fit(X_train, y_train)
lr_train_pred = lr.predict(X_train)
lr_test_pred = lr.predict(X_test)

# accuracy
lr_train_acc = np.mean(lr_train_pred == y_train)
lr_test_acc = np.mean(lr_test_pred == y_test)

# precision
lr_train_prec = precision_score(y_train, lr_train_pred, average='weighted', zero_division=0)
lr_test_prec = precision_score(y_test, lr_test_pred, average='weighted', zero_division=0)



model_results.loc[model_results['Model'] == 'LR', 'Accuracy_Train'] = lr_train_acc
model_results.loc[model_results['Model'] == 'LR', 'Accuracy_Test'] = lr_test_acc
model_results.loc[model_results['Model'] == 'LR', 'Precision_Train'] = lr_train_prec
model_results.loc[model_results['Model'] == 'LR', 'Precision_Test'] = lr_test_prec


# # K-nearest neighbors 

# ## Hyperparameter turning

# In[25]:


allks = range(1, 31)

train_acc = []
test_acc = []
train_precision = []
test_precision = []


for k in allks:

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_yhat = knn.predict(X_train)
    test_yhat = knn.predict(X_test)
    

    train_acc.append(accuracy_score(y_train, train_yhat))
    test_acc.append(accuracy_score(y_test, test_yhat))
    train_precision.append(precision_score(y_train, train_yhat))
    test_precision.append(precision_score(y_test, test_yhat))


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
ax1.plot(allks, train_acc, marker="o", color="green", lw=3, label="Training Accuracy")
ax1.plot(allks, test_acc, marker="o", color="steelblue", lw=3, label="Test Accuracy")
ax1.set_xlabel("Number of Neighbors (k)", fontsize=16)
ax1.set_ylabel("Accuracy", fontsize=16)
ax1.set_xticks(range(1, 31, 2))
ax1.grid(alpha=0.25)
ax1.legend(fontsize=14)
ax1.set_title("Accuracy vs. Number of Neighbors", fontsize=16)

ax2.plot(allks, train_precision, marker="o", color="green", lw=3, label="Training Precision")
ax2.plot(allks, test_precision, marker="o", color="steelblue", lw=3, label="Test Precision")
ax2.set_xlabel("Number of Neighbors (k)", fontsize=16)
ax2.set_ylabel("Precision", fontsize=16)
ax2.set_xticks(range(1, 31, 2))
ax2.grid(alpha=0.25)
ax2.legend(fontsize=14)
ax2.set_title("Precision vs. Number of Neighbors", fontsize=16)

plt.show()


# ## Final K-nearest neighbors Model
# 
# The improvement in Accuracy (in the test data) stop around k=12. However the Precisoin contuines to improve right to the end of tested K values. For the final model we will use k=30.

# In[17]:


knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
knn_train_pred = knn.predict(X_train)
knn_test_pred = knn.predict(X_test)

# accuracy
knn_train_acc = np.mean(knn_train_pred == y_train)
knn_test_acc = np.mean(knn_test_pred == y_test)

# precision
knn_train_prec = precision_score(y_train, knn_train_pred, average='weighted', zero_division=0)
knn_test_prec = precision_score(y_test, knn_test_pred, average='weighted', zero_division=0)


model_results.loc[model_results['Model'] == 'KNN', 'Accuracy_Train'] = knn_train_acc
model_results.loc[model_results['Model'] == 'KNN', 'Accuracy_Test'] = knn_test_acc
model_results.loc[model_results['Model'] == 'KNN', 'Precision_Train'] = knn_train_prec
model_results.loc[model_results['Model'] == 'KNN', 'Precision_Test'] = knn_test_prec


# # Random Forest

# ## Hyperparameter turning

# In[33]:


max_depths = range(1, 21)  
max_leaf_nodes_list = [None, 10, 20, 50]  

train_acc = {leaf: [] for leaf in max_leaf_nodes_list}
test_acc = {leaf: [] for leaf in max_leaf_nodes_list}
train_precision = {leaf: [] for leaf in max_leaf_nodes_list}
test_precision = {leaf: [] for leaf in max_leaf_nodes_list}


for max_depth in max_depths:
    for max_leaf_nodes in max_leaf_nodes_list:
        
        rf = RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        rf.fit(X_train, y_train)
        
    
        train_yhat = rf.predict(X_train)
        test_yhat = rf.predict(X_test)
        
        train_acc[max_leaf_nodes].append(accuracy_score(y_train, train_yhat))
        test_acc[max_leaf_nodes].append(accuracy_score(y_test, test_yhat))

        train_prec = precision_score(y_train, train_yhat, average='weighted', zero_division=1)
        test_prec = precision_score(y_test, test_yhat, average='weighted', zero_division=1)
        
        train_precision[max_leaf_nodes].append(train_prec)
        test_precision[max_leaf_nodes].append(test_prec)



# In[34]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))


for max_leaf_nodes in max_leaf_nodes_list:
    ax1.plot(max_depths, train_acc[max_leaf_nodes], marker="o", lw=3, label=f"Max Leaf Nodes={max_leaf_nodes}")
ax1.set_xlabel("Max Depth", fontsize=16)
ax1.set_ylabel("Accuracy", fontsize=16)
ax1.set_xticks(range(1, 21, 2))
ax1.grid(alpha=0.25)
ax1.legend(title="Max Leaf Nodes", fontsize=12)
ax1.set_title("Train Accuracy vs Max Depth (Different Max Leaf Nodes)", fontsize=16)

for max_leaf_nodes in max_leaf_nodes_list:
    ax2.plot(max_depths, test_acc[max_leaf_nodes], marker="o", lw=3, label=f"Max Leaf Nodes={max_leaf_nodes}")
ax2.set_xlabel("Max Depth", fontsize=16)
ax2.set_ylabel("Accuracy", fontsize=16)
ax2.set_xticks(range(1, 21, 2))
ax2.grid(alpha=0.25)
ax2.legend(title="Max Leaf Nodes", fontsize=12)
ax2.set_title("Test Accuracy vs Max Depth (Different Max Leaf Nodes)", fontsize=16)


for max_leaf_nodes in max_leaf_nodes_list:
    ax3.plot(max_depths, train_precision[max_leaf_nodes], marker="o", lw=3, label=f"Max Leaf Nodes={max_leaf_nodes}")
ax3.set_xlabel("Max Depth", fontsize=16)
ax3.set_ylabel("Precision", fontsize=16)
ax3.set_xticks(range(1, 21, 2))
ax3.grid(alpha=0.25)
ax3.legend(title="Max Leaf Nodes", fontsize=12)
ax3.set_title("Train Precision vs Max Depth (Different Max Leaf Nodes)", fontsize=16)


for max_leaf_nodes in max_leaf_nodes_list:
    ax4.plot(max_depths, test_precision[max_leaf_nodes], marker="o", lw=3, label=f"Max Leaf Nodes={max_leaf_nodes}")
ax4.set_xlabel("Max Depth", fontsize=16)
ax4.set_ylabel("Precision", fontsize=16)
ax4.set_xticks(range(1, 21, 2))
ax4.grid(alpha=0.25)
ax4.legend(title="Max Leaf Nodes", fontsize=12)
ax4.set_title("Test Precision vs Max Depth (Different Max Leaf Nodes)", fontsize=16)

plt.tight_layout()
plt.show()


# ## Final Random Forest Model
# 
# Improvements on preformace peaked at around 11 max depth. No limit on the max leaf nodes preformed best in all four groups above. 

# In[20]:


RF = RandomForestClassifier(max_depth=None, max_leaf_nodes=11)
RF.fit(X_train, y_train)
RF_train_pred = RF.predict(X_train)
RF_test_pred = RF.predict(X_test)

# accuracy
RF_train_acc = np.mean(RF_train_pred == y_train)
RF_test_acc = np.mean(RF_test_pred == y_test)

# precision
RF_train_prec = precision_score(y_train, RF_train_pred, average='weighted', zero_division=0)
RF_test_prec = precision_score(y_test, RF_test_pred, average='weighted', zero_division=0)

model_results.loc[model_results['Model'] == 'RF', 'Accuracy_Train'] = RF_train_acc
model_results.loc[model_results['Model'] == 'RF', 'Accuracy_Test'] = RF_test_acc
model_results.loc[model_results['Model'] == 'RF', 'Precision_Train'] = RF_train_prec
model_results.loc[model_results['Model'] == 'RF', 'Precision_Test'] = RF_test_prec


# # Results and Analysis
# 
# Random Forest had the best performance overall.
# 
# #### Logistic Regression
# 
# Tuning the parameter (C) did not change accuracy or precision. The model’s performance stayed the same, suggesting it was not a good fit for this dataset.
# 
# #### K-Nearest Neighbors (KNN)
# 
# Accuracy: Increased up to k=12, then leveled off.
# 
# Precision: Improved steadily up to the maxiumum.
# 
# This shows that higher k values help precision but have less effect on accuracy after a point.
# 
# #### Random Forest
# 
# Random Forest performed the best:
# 
# Maximum Depth: Accuracy stopped improving at a depth of 7. Precision peaked at a depth of 11.
# 
# Maximum Leaf Nodes: No limit on leaf nodes (max_leaf_nodes=None) gave the best results.
# 
# Random Forest captured complex patterns better than the other models.
# 
# #### Comparative Performance
# 
# The models ranked as follows:
# 
# Random Forest: Best accuracy and precision.
# 
# KNN: Good precision at higher k values.
# 
# Logistic Regression: Least effective with no improvement during tuning.
# 
# 

# In[21]:


# ploting results
x = np.arange(len(model_results))  
width = 0.35 

fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

ax[0].bar(x - width/2, model_results['Accuracy_Train'], width, label='Train', color='skyblue')
ax[0].bar(x + width/2, model_results['Accuracy_Test'], width, label='Test', color='steelblue')
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Models')
ax[0].set_ylabel('Scores')
ax[0].set_xticks(x)
ax[0].set_xticklabels(model_results['Model'])
ax[0].legend()

ax[1].bar(x - width/2, model_results['Precision_Train'], width, label='Train', color='skyblue')
ax[1].bar(x + width/2, model_results['Precision_Test'], width, label='Test', color='steelblue')
ax[1].set_title('Model Precision')
ax[1].set_xlabel('Models')
ax[1].set_ylabel('Scores')
ax[1].set_xticks(x)
ax[1].set_xticklabels(model_results['Model'])
ax[1].tick_params(axis='y', labelleft=True)
ax[1].legend()


plt.show()


# # Discussion and Conclusion

# Random Forest worked best because it handles complex relationships in data. Logistic Regression was limited because it assumes linear relationships. KNN performed moderately well but needed larger k values for better precision.
# 
# Choosing the right model and tuning it is important for good results. While simpler models are easier to use, complex models like Random Forest often give better performance with complex data.
# 
# To improve the models, we can consider:
# 
#     Feature Engineering (Creating new features or selecting the most relevant features)
#     Cross-Validation
#     Oversampling/Undersampling
#     
