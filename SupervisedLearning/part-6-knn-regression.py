import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
sensus = {
	'tinggi' : [150, 170, 183, 191, 155, 163, 180, 158, 178],
	'jk' : ['pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita'],
	'berat' : [64, 86, 84, 80, 49, 59, 67, 54, 67]
}


sensus_df = pd.DataFrame(sensus)
print(sensus_df)

x_train = np.array(sensus_df[['tinggi', 'jk']])
y_train = np.array(sensus_df['berat'])

print(f'x_train : \n {x_train} \n')
print(f'y_train : \n {y_train} \n')

x_train_transposed = np.transpose(x_train)  # transpose mengubah posisi baris menjadi kolom (sebaliknya)

print(f'x_train : \n {x_train} \n')
print(f'x_train_transposed : \n {x_train_transposed} \n')

lb = LabelBinarizer()
jk_binarised = lb.fit_transform(x_train_transposed[1])

print(f'jk : {x_train_transposed[1]}\n')
print(f'jk_binarised : {jk_binarised}\n')

jk_binarised = jk_binarised.flatten()
print(jk_binarised)

x_train_transposed[1] = jk_binarised
x_train = x_train_transposed.transpose()

print(f'x_train_transposed : \n {x_train_transposed} \n')
print(f'x_train : \n {x_train} \n')

k = 3
model = KNeighborsRegressor(n_neighbors = k)
model.fit(x_train, y_train)

x_new = np.array([[155, 1]])
print(x_new)

y_pred = model.predict(x_new)
print(y_pred)



















# ==== Mengatur Peforma ====
x_test = np.array([[168, 0], [180, 0], [160, 1], [169, 1]])
y_test = np.array([65, 96, 52, 67])

print(f'x_test : \n {x_test} \n')
print(f'y_test : \n {y_test} \n')

y_pred = model.predict(x_test)
print(y_pred)

# R kuadrat
r_squared = r2_score(y_test, y_pred)
print(f'r_squared : {r_squared} \n')

# Mean Absolute Error (Deviation)
MAE = mean_absolute_error(y_test, y_pred)
print(f'MAE : {MAE} \n')

# Mean Squared Error
MSE = mean_squared_error(y_test, y_pred)
print(f'MSE : {MSE} \n')





# === PERMASALAHAN ===
x_train = np.array([[1700, 0], [1600, 1]])	# satuan tinggi (mm)
x_new = np.array([[1640, 0]])

[euclidean(x_new[0], d) for d in x_train]


x_train = np.array([[1.7, 0], [1.6, 1]])	# satuan tinggi (cm)
x_new = np.array([[1.64, 0]])

[euclidean(x_new[0], d) for d in x_train]




# === Mengatasi Permasalahan ===
ss = StandardScaler()

# tinggi mm
x_train = np.array([[1700, 0], [1600,1]])
x_train_scaled = ss.fit_transform(x_train)
print(f'x_train_scaled : \n {x_train_scaled} \n')


x_new = np.array([[1640, 0]])
x_new_scaled = ss.transform(x_new)
print(f'x_new_scaled : \n {x_new_scaled} \n')

jarak = [euclidean(x_new_scaled[0], d) for d in x_train_scaled]
print(f'jarak : {jarak} \n')

# tinggi m
x_train = np.array([[1.7, 0], [1.6,1]])
x_train_scaled = ss.fit_transform(x_train)
print(f'x_train_scaled : \n {x_train_scaled} \n')


x_new = np.array([[1.64, 0]])
x_new_scaled = ss.transform(x_new)
print(f'x_new_scaled : \n {x_new_scaled} \n')

jarak = [euclidean(x_new_scaled[0], d) for d in x_train_scaled]
print(f'jarak : {jarak} \n')




x_train = np.array([[158, 0], [170, 0], [183, 0], [191, 0], [155, 1], [163, 1], [180, 1], [158, 1], [170, 1]])
y_train = np.array([64, 86, 84, 80, 49, 59, 67, 54, 67])

x_test = np.array([[168, 0], [180, 0], [160, 0], [169, 1]])
y_test = np.array([65, 96, 52, 67])

x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

print(f'x_train_scaled : \n {x_train_scaled} \n')
print(f'x_test_scaled : \n {x_test_scaled} \n')
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)


MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
print(f'MAE : {MAE}')
print(f'MSE : {MSE}')


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""