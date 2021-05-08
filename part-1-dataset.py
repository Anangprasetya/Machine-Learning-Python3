from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
print(iris)
print(iris.keys())
print(iris.DESCR)

x = iris.data 
print(x.shape)
print(x)

y = iris.target
print(y.shape)
print(y)

feature_name = iris.feature_names
print(feature_name)

target_names = iris.target_names
print(target_names)


# Memvisualisasikan data
x = x[:, :2]
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5

plt.scatter(x[:, 0], x[:, 1], c = y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
plt.show()




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

print(f'x_train : {x_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_train : {y_train.shape}')
print(f'y_test : {y_test.shape}')



iris = load_iris(as_frame = True)

iris_feature_df = iris.data
print(iris_feature_df)









"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""