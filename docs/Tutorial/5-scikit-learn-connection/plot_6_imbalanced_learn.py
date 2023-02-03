"""
===========================
Work with imbalanced-learn
===========================
``Imbalanced-learn`` is an open source, MIT-licensed library relying on scikit-learn 
and provides tools when dealing with classification with imbalanced classes. In this tutorial, 
we will show how to combine ``abess.linear.LogisticRegression`` and ``imbalanced-learn`` to 
handle a imbalanced binary classification task.
"""

#%%
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from abess.linear import LogisticRegression
from abess.datasets import make_glm_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours


#%%
# Synthetic data
# ---------------


#%%
# Generate imbalanced dataset (X, y). Here, we use ``make_glm_data`` to generate a balanced 
# binary dataset ``data`` and then drop 90% of positive samples. Thus, the imbalance ratio of
# our example is around 10:1.
n, p, k = 5000, 2000, 10
random_state = 12345
np.random.seed(random_state)
data = make_glm_data(n=n, p=p, k=k, family='binomial')
idx0 = np.where(data.y == 0)[0]  # index of negative sample
idx1 = np.where(data.y == 1)[0]  # index of positive sample
idx = np.array(list(set(idx0).union(set(idx1[:int(n/20)]))))
X, y = data.x[idx], data.y[idx]
print('Generated dataset has {} positive samples and {} negative samples.'.format(np.sum(y==1), np.sum(y==0)))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
print('Train size: {}, Test size: {}.'.format(len(y_train), len(y_test)))


#%%
# Base estimator
# ---------------
model = LogisticRegression(support_size=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# Over-sampling
# --------------


#%%
# RandomOverSampler
ros = RandomOverSampler()
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
model = LogisticRegression(support_size=k)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print('Resampled size: ', len(y_train_resampled))
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# SMOTE

X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
model = LogisticRegression(support_size=k)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print('Resampled size: ', len(y_train_resampled))
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# ADASYN

X_train_resampled, y_train_resampled = ADASYN().fit_resample(X_train, y_train)
model = LogisticRegression(support_size=k)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print('Resampled size: ', len(y_train_resampled))
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# Under-sampling
# ----------------


#%%
# RandomUnderSampler

rus = RandomUnderSampler()
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
model = LogisticRegression(support_size=k)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print('Resampled size: ', len(y_train_resampled))
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# EditedNearestNeighbours

enn = EditedNearestNeighbours(kind_sel='all')
X_train_resampled, y_train_resampled = enn.fit_resample(X_train, y_train)
model = LogisticRegression(support_size=k)
model.fit(X_train_resampled, y_train_resampled)
y_pred = model.predict(X_test)
print('Resampled size: ', len(y_train_resampled))
print('Balanced accuracy score: ', balanced_accuracy_score(y_test, y_pred).round(3))


#%%
# Pipeline
# ---------

#%%
# In the following, we show how to construct a pipeline.
# Note that pipeline implemented by sklearn requires that all intermediate 
# estimators must be transformers.
# However, resamplers in imblearn are not transformers.
# Instead, we explicitly use pipeline implemented by imblearn here.

from imblearn.pipeline import Pipeline as imbPipeline
resamplers = {  
                'RandomOverSampler': RandomOverSampler, 
                'SMOTE': SMOTE, 
                'ADASYN': ADASYN, 
                'RandomUnderSampler': RandomUnderSampler, 
                'EditedNearestNeighbours': EditedNearestNeighbours
            }
for name in resamplers.keys():
    resampler = resamplers[name]()
    estimators = [('resampler', resampler), ('clf', LogisticRegression(support_size=k))]
    pipe = imbPipeline(estimators)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print('{}: {}'.format(name, balanced_accuracy_score(y_test, y_pred).round(3)) )


# %%
# sphinx_gallery_thumbnail_path = 'Tutorial/figure/imbalanced-learn.png'