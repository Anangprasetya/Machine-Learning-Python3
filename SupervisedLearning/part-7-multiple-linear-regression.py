import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
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

x_train = np.array(train_pizza_df[['diameter', 'n_topping']])
y_train = np.array(train_pizza_df['harga'])

print(f'x_train : \n {x_train} \n')
print(f'y_train : \n {y_train} \n')

x_test = np.array(test_pizza_df[['diameter', 'n_topping']])
y_test = np.array(test_pizza_df['harga'])

print(f'x_test : \n {x_test} \n')
print(f'y_test : \n {y_test} \n')

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f'r_squared : {r2_score(y_test, y_pred)}')


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""