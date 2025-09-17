import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

veri = pd.read_csv("/Users/mehmet/Downloads/G-s-Kanseri-master/breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace=True)
veri = veri.drop(['id'], axis=1)

y = np.array(veri.benormal)
x = np.array(veri.drop(['benormal'], axis=1))

imp = SimpleImputer(missing_values=-99999, strategy="mean")
x = imp.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

tahmin = KNeighborsClassifier()

tahmin.fit(X_train, y_train)
basaria = tahmin.score(X_test, y_test)

print("Yüzde", basaria*100, "oranında:")
a= np.array([2,3,4,5,6,7,9,7,1]).reshape(1, -1)
print(tahmin.predict(a))
