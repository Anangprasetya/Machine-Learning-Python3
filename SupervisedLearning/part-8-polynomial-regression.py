import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

pizza = {
	'diameter' : [6, 8, 10, 14, 18],
	'n_topping' : [2, 1, 0, 2, 0],
	'harga' : [7, 9, 13, 17.5, 18]
}

train_pizza_df = pd.DataFrame(pizza)
print(train_pizza_df)

pizza = {
	'diameter' : [8, 9, 11, 16, 12],
	'n_topping' : [2, 0, 2, 2, 0],
	'harga' : [11, 8.5, 15, 18, 11]
}

test_pizza_df = pd.DataFrame(pizza)
print(test_pizza_df)

x_train = np.array(train_pizza_df['diameter']).reshape(-1, 1)
y_train = np.array(train_pizza_df['harga'])

print(f'x_train : \n {x_train} \n')
print(f'y_train : \n {y_train} \n')

quadratic_feature = PolynomialFeatures(degree = 2)
x_train_quadratic = quadratic_feature.fit_transform(x_train)

print(f'x_train_quadratic : \n {x_train_quadratic} \n')

model = LinearRegression()
model.fit(x_train_quadratic, y_train)

x_vis = np.linspace(0, 25, 100).reshape(-1, 1)
x_vis_quadratic = quadratic_feature.transform(x_vis)
y_vis_quadratic = model.predict(x_vis_quadratic)

plt.scatter(x_train, y_train)
plt.plot(x_vis, y_vis_quadratic, 'r-')

plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diamer inc')
plt.ylabel('Harga dollar')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)
plt.show()


plt.scatter(x_train, y_train)


#Linear
model = LinearRegression()
model.fit(x_train, y_train)
x_vis = np.linspace(0, 25, 100).reshape(-1, 1)
y_vis = model.predict(x_vis)
plt.plot(x_vis, y_vis, '--r', label = 'linear')



#Quadratic
quadratic_feature = PolynomialFeatures(degree = 2)
x_train_quadratic = quadratic_feature.fit_transform(x_train)
model = LinearRegression()
model.fit(x_train_quadratic, y_train)
x_vis_quadratic = quadratic_feature.transform(x_vis)
y_vis = model.predict(x_vis_quadratic)
plt.plot(x_vis, y_vis, '--g', label = 'quadratic')

cubic_feature = PolynomialFeatures(degree = 3)
x_train_cubic = cubic_feature.fit_transform(x_train)
model = LinearRegression()
model.fit(x_train_cubic, y_train)
x_vis_cubic = cubic_feature.transform(x_vis)
y_vis = model.predict(x_vis_cubic)
plt.plot(x_vis, y_vis, '--y', label = 'cubic')

plt.title('Perbandingan')
plt.xlabel('diameter')
plt.ylabel('harga')
plt.legend()
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)
plt.show()


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""