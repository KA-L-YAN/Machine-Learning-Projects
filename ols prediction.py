import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('txtbased class4Solo.csv')


print(data.shape)

from sklearn.metrics import mean_squared_error
print(data.info())

import sys


Y=data.iloc[:,11:12].values
X=data.iloc[:,3:9].values
print(Y)
print(X)

import statsmodels.api as sm
one=np.ones((177,1),dtype=int)
X=np.append(arr=one,values=X,axis=1)
print(data.head())
print(data.shape)
print(data.info())
X_new = np.array(X[:, [0,1,3,5]], dtype=float)
obj = sm.OLS(endog = Y, exog = X_new).fit()
print(obj.summary())


from statsmodels.formula.api import ols
data1 = {"x1": X_new, "y": Y}
res = ols("Y ~ X + np.sin(X) + I((X-5)**2)", data=data).fit()
print(res.params)
pred=res.predict(exog=dict(x1=Y))
print(pred)
print(type(pred))
df = pd.DataFrame(pred, columns = ['pred'])
df.to_csv('class4solools.csv',index=False)








sys.stdout=open("txtbased class4Solo.txt","w")
print(obj.summary())
sys.stdout.close()


