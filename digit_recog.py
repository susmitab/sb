import pandas 
from sklearn.ensemble import RandomForestClassifier


digit =  pandas.read_csv("train.csv")
digit_test = pandas.read_csv("test.csv").values

label = digit[[0]].values.ravel()
pixel = digit.iloc[:,1:].values


alg = RandomForestClassifier(n_estimators=100)

alg.fit(pixel,label)
score = alg.score(pixel,label)
print score

predictions = alg.predict(digit_test)

submit = pandas.DataFrame({"ImageId":range(1,len(digit_test)+1), "Label": predictions})

submit.to_csv("kaggle_digit.csv", index=False)