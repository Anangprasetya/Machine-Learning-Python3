from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
print("model : ",model)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi adalah  : ", acc)
print("Akurasi adalah  : %0.2f" % acc)


data_baru = [[5, 5, 3, 2],
			 [2, 4, 3, 5]]

preds = model.predict(data_baru)
print(preds)

pred_sprecies = [iris.target_names[p] for p in preds ]
print(pred_sprecies)

joblib.dump(model, 'iris_classifier_knn.joblib')

production_model = joblib.load('iris_classifier_knn.joblib')
print("production_model : ",production_model)


print(production_model.predict([[5, 5, 2, 3]]))




"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""