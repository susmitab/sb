import pandas
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model.logistic import 
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold

digit =  pandas.read_csv("train.csv")
digit_test = pandas.read_csv("test.csv").values

label = digit[[0]].values.ravel()
pixel = digit.iloc[:,1:].values


alg = RandomForestClassifier(n_estimators=100)

alg.fit(pixel,label)
score = alg.score(pixel,label)
print score

#alg.fit(digit[predictors],digit["label"])
#
predictions = alg.predict(digit_test)

submit = pandas.DataFrame({"ImageId":range(1,len(digit_test)+1), "Label": predictions})

submit.to_csv("kaggle_digit.csv", index=False)