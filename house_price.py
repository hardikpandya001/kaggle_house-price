import pandas as pd
desired_width=200
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',100)
import seaborn as sns
import matplotlib.pyplot as plt
train=pd.read_csv('C:\\Users\\hardi\\Desktop\\house_price\\train.csv')

test=pd.read_csv('C:\\Users\\hardi\\Desktop\\house_price\\test.csv')
print(train.columns.shape)
print(test.columns.shape)
#print(train.columns)
#print(test.columns.shape)
#print(train.head(2))
#print(train.info())

train.dropna(axis=1,inplace=True)
test.dropna(axis=1,inplace=True)
print(train.columns.shape)
print(test.columns.shape)
train.dropna(subset=['SalePrice'],inplace=True)
df=pd.DataFrame(train)
x_train= train.drop(['SalePrice','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','SaleType','SaleCondition','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','Heating','HeatingQC','CentralAir','KitchenQual','Functional','PavedDrive','YrSold','Id', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFullBath', 'BsmtHalfBath','GarageCars', 'GarageArea','Street'],axis=1)

test_x=test.drop(['Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1' ,'Condition2','BldgType','HouseStyle','SaleCondition','RoofStyle','RoofMatl','ExterQual','ExterCond','Foundation','Heating','HeatingQC', 'CentralAir','PavedDrive','YrSold','Id','Electrical','Street'],axis=1)
y_train=train['SalePrice']
print(x_train.columns.shape)
print(test_x.columns.shape)
print(test_x.head(5))

"""
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.pairplot(data=df,hue='SalePrice')
plt.show()
print(train.columns)
print(x_train.columns.shape)
print(test.columns.shape)
print(x_train.columns)
print(test.columns)
"""

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
predictions=model.predict(test_x)
submission=pd.DataFrame({'Id':test['Id'],'SalePrice':predictions})
filename = 'HousePrice.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)