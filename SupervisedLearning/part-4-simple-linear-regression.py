import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pizza = {
	'diameter':[6, 8, 10, 14, 18],
	'harga':[7, 9, 13, 17.5, 18]
} 

pizza_df = pd.DataFrame(pizza)
print(pizza_df)

pizza_df.plot(kind = 'scatter', x = 'diameter', y = 'harga')
#paramter pertama untuk scatter, paramter ke dua untuk sumbu x, parameter ke 3 untuk sumbu y

plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')
plt.xlim(0, 25)	#untuk mengatur jangkauan sumbu x
plt.ylim(0, 25)	# untuk mengatur jangkuan sumbu y
plt.grid(True)	# aktifkan grid
plt.show()


X = np.array(pizza_df['diameter'])
y = np.array(pizza_df['harga'])

print(f'x: {X}')
print(f'y: {y}')

#ubah dari 1 dimensi menjadi 2 dimensi
X = X.reshape(-1, 1)
print(X.shape)
print(X)


model = LinearRegression()
model.fit(X, y)

x_vis = np.array([0, 25]).reshape(-1, 1)  #membuat nilai min dan max
y_vis = model.predict(x_vis)


plt.scatter(X, y)
plt.plot(x_vis, y_vis, '-r')


plt.title('Perbandingan Diameter dan Harga Pizza')
plt.xlabel('Diameter (inch)')
plt.ylabel('Harga (dollar)')
plt.xlim(0, 25)
plt.ylim(0, 25)
plt.grid(True)
plt.show()

print(f'intercept : {model.intercept_}')	# titik sumbu y yang di kenai garis linear ketika x = 0
print(f'slope : {model.coef_}') 			# kemiringan garis (gradien)



# == Tentang intercept dan slope ==

print(f'x:\n{X}\n')
print(f'x flatten: {X.flatten()}\n')
print(f'y: {y}')


# Mencari slope
# Menhitung nilai variance
variance_x = np.var(X.flatten(), ddof = 1)  # degre of readem
print(f'variance : {variance_x}') 


# Menghitung covariance
np.cov(X.flatten(), y)
covariance_xy = np.cov(X.flatten(), y)[0][1]
print(f'covariance : {covariance_xy}')

slope = covariance_xy / variance_x
print(f'slope : {slope}')


# Mencari intercept
intercept = np.mean(y) - slope * np.mean(X)
print(f'intercept : {intercept}')




# Prediksi harga pizza
diameter_pizza = np.array([12, 20, 23]).reshape(-1, 1)
print(diameter_pizza)

prediksi_harga = model.predict(diameter_pizza)
print(prediksi_harga)

for dmtr, hrg in zip(diameter_pizza, prediksi_harga):
	print(f'Diameter: {dmtr} prediksi harga: {hrg}')






# === Untuk mengetes permorfa ===
x_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
y_train = np.array([7, 9, 13, 17.5, 18])

x_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = np.array([11, 8.5, 15, 18, 11])


model = LinearRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)

r_squared = r2_score(y_test, y_pred)

print(f'R-squared : {r_squared}')


# mencari SSres
ss_res = sum([(y_i - model.predict(x_i.reshape(-1, 1))[0])**2
			for x_i, y_i in zip(x_test, y_test)])

print(f'ss_res : {ss_res}')



mean_y = np.mean(y_test)
ss_tot = sum([(y_i - mean_y)**2 for y_i in y_test])

print(f'ss_tot: {ss_tot}')



r_squared = 1 - (ss_res / ss_tot)
print(f'R-squared : {r_squared}')


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""