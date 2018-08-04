#necessary imports
################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
################################################################################

# path = "adult.data"
path = "/home/arpit/learning/machine learning/Backyard Data Science/adult.data"
###############################################################################
col = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","result"]
df = pd.read_csv(path,header=None)
# print df.head()
# print df.shape
# print df.isnull().any()
# df = df.as_matrix()
data = pd.DataFrame()
i=0
for o in col:
    data[o] = df[i]
    i = i+1


################################################################################
#inspecting the data
print (data.head())

data.drop('fnlwgt',axis=1)
#for relation in  numeric data's
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,scale
import seaborn as sns
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(data["result"])
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X_numeric = data.select_dtypes(include=numerics)
print X_numeric.head()

#############################################
# std and mean before scaling
print (X_numeric.std())
print ((X_numeric).mean())

###################################
# Scaling
X_numeric = scale(X_numeric)

######################################################
# std and mean after scaling
print (X_numeric.std())
print ((X_numeric).mean())


############################################################################
# Makinng Correation matrix
df2 =  pd.DataFrame(data=X_numeric)
df2["result"] = y
corr = df2.corr()
# sns.heatmap(corr)
# plt.show()

###################################################
X_numeric = np.array(X_numeric)
# print (newdf)

inspect =  data.head()

################################################################################
#adding X and y

X = data.iloc[:, :-1].values
y = data.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label encoding y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

labelencoder_X = LabelEncoder()
LISTY = 1,3,5,6,7,8,9,13

X_category = data.iloc[:, [1,3,5,6,7,8,9,13]].values

#label encoding X
labelencoder_x = LabelEncoder()
X_category[:, 0] = labelencoder_x.fit_transform(X_category[:,0])
X_category[:, 1] = labelencoder_x.fit_transform(X_category[:,1])
X_category[:, 2] = labelencoder_x.fit_transform(X_category[:,2])
X_category[:, 3] = labelencoder_x.fit_transform(X_category[:,3])
X_category[:, 4] = labelencoder_x.fit_transform(X_category[:,4])
X_category[:, 5] = labelencoder_x.fit_transform(X_category[:,5])
X_category[:, 6] = labelencoder_x.fit_transform(X_category[:,6])
X_category[:, 7] = labelencoder_x.fit_transform(X_category[:,7])

################################################################################
#One hot encoding Y
onehotencoder = OneHotEncoder(categorical_features = [0])
X_category = onehotencoder.fit_transform(X_category).toarray()

onehotencoder = OneHotEncoder(categorical_features = [9])
X_category = onehotencoder.fit_transform(X_category).toarray()


onehotencoder = OneHotEncoder(categorical_features = [25])
X_category = onehotencoder.fit_transform(X_category).toarray()

onehotencoder = OneHotEncoder(categorical_features = [32])
X_category = onehotencoder.fit_transform(X_category).toarray()

onehotencoder = OneHotEncoder(categorical_features = [47])
X_category = onehotencoder.fit_transform(X_category).toarray()

onehotencoder = OneHotEncoder(categorical_features = [53])
X_category = onehotencoder.fit_transform(X_category).toarray()


onehotencoder = OneHotEncoder(categorical_features = [59])
X_category = onehotencoder.fit_transform(X_category).toarray()

################################################################################
#concatinating
X = np.concatenate([X_category,X_numeric],axis=1)

################################################################################
#test train split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 325)


################################################################################
#logistic
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#################################################################################3
# Confution matrix and acc.
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(cm)
##################################################################

import xgboost as xgb   
# fit model no training data
gbm = xgb.XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05).fit(X_train, y_train)
y_pred = gbm.predict(X_test)
# Confution matrix and ac5.
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(cm)
