#!/usr/bin/env python
# coding: utf-8

# **Run the following two cells before you begin.**

# In[1]:


get_ipython().run_line_magic('autosave', '10')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('cleaned_data.csv')


# **Run the following 3 cells to create a list of features, create a train/test split, and instantiate a random forest classifier.**

# In[3]:


features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
features_response


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[features_response[:-1]].values,
    df['default payment next month'].values,
    test_size=0.2, random_state=24
)


# In[5]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
     criterion='gini', max_depth=3,
    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
    random_state=4, verbose=0, warm_start=False, class_weight=None
)


# **Create a dictionary representing the grid for the `max_depth` and `n_estimators` hyperparameters that will be searched. Include depths of 3, 6, 9, and 12, and 10, 50, 100, and 200 trees.**

# In[6]:


grid= {'max_depth':[3,6,9,12],
 'n_estimators':[10,50,100,200]}
grid


# ________________________________________________________________
# **Instantiate a `GridSearchCV` object using the same options that we have previously in this course, but with the dictionary of hyperparameters created above. Set `verbose=2` to see the output for each fit performed.**

# In[7]:


from sklearn.model_selection import GridSearchCV
cv_rf = GridSearchCV(rf, param_grid=grid, scoring='roc_auc',
                  n_jobs=None, iid=False, refit=True, cv=4, verbose=2,
                  pre_dispatch=None, error_score=np.nan, return_train_score=True)


# ____________________________________________________
# **Fit the `GridSearchCV` object on the training data.**

# In[9]:


best_model=cv_rf.fit(X_train,y_train)


# In[10]:


yhat=cv_rf.predict(X_test)
len(yhat)


# ___________________________________________________________
# **Put the results of the grid search in a pandas DataFrame.**

# In[11]:


cv_rf_results_df=pd.DataFrame(cv_rf.cv_results_)
cv_rf_results_df


# **Find the best hyperparameters from the cross-validation.**

# In[ ]:


cv_rf.best_params_


# ________________________________________________________________________________________________________
# **Create a `pcolormesh` visualization of the mean testing score for each combination of hyperparameters.**
# 
# <details>
#     <summary>Hint:</summary>
#     Remember to reshape the values of the mean testing scores to be a two-dimensional 4x4 grid.
# </details>

# In[12]:


# Create a 5x5 gridv
xx_rf, yy_rf = np.meshgrid(range(5), range(5))


# In[13]:


# Set color map to `plt.cm.jet`
cm_rf = plt.cm.jet


# In[14]:


ax_rf = plt.axes()
pcolor_graph = ax_rf.pcolormesh(xx_rf, yy_rf, cv_rf_results_df['mean_test_score'].values.reshape((4,4)), cmap=cm_rf)
plt.colorbar(pcolor_graph, label='Average testing ROC AUC')
ax_rf.set_aspect('equal')
ax_rf.set_xticks([0.5, 1.5, 2.5, 3.5])
ax_rf.set_yticks([0.5, 1.5, 2.5, 3.5])
ax_rf.set_xticklabels([str(tick_label) for tick_label in grid['n_estimators']])
ax_rf.set_yticklabels([str(tick_label) for tick_label in grid['max_depth']])
ax_rf.set_xlabel('Number of trees')
ax_rf.set_ylabel('Maximum depth')


# ________________________________________________________________________________________________________
# **Conclude which set of hyperparameters to use.**

# In[15]:


feat_imp_df_act = pd.DataFrame({
    'Feature name':features_response[:-1],
    'Importance':cv_rf.best_estimator_.feature_importances_
})
feat_imp_df_act


# In[16]:


# Sort values by importance
feat_imp_df_act = pd.DataFrame({
    'Feature name':features_response[:-1],
    'Importance':cv_rf.best_estimator_.feature_importances_
})


# In[17]:


feat_imp_df_act.sort_values('Importance', ascending=False)


# In[18]:


import pickle
file = open('loan.pkl','wb')# open a file where you want to store the data
pickle.dump(cv_rf,file)# dump information to that file
model = open('loan.pkl','rb')
forest =pickle.load(model)


# In[29]:



prediction= forest.predict(X_test)
print(prediction)


# In[35]:


np.array(X_test[0,:])


# In[ ]:




