import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef



sensus = {
	'tinggi' : [150, 170, 183, 191, 155, 163, 180, 158, 178],
	'berat' : [64, 86, 84, 80, 49, 59, 67, 54, 67],
	'jk' : ['pria', 'pria', 'pria', 'pria', 'wanita', 'wanita', 'wanita', 'wanita', 'wanita']
}

sensus_df = pd.DataFrame(sensus)
# print(sensus_df)


fig, ax = plt.subplots()
for jk, d in sensus_df.groupby('jk'):
	ax.scatter(d['tinggi'], d['berat'], label = jk)


plt.legend(loc = 'upper left')  # loc -> mengatur posisi di atas kiri
plt.title('Sebaran Data tinggi badan, berat badan dan jenis jelamin')
plt.xlabel('tinggi (cm)')
plt.ylabel('berat (kg)')
plt.grid(True)
plt.show()




x_train = np.array(sensus_df[['tinggi', 'berat']])
y_train = np.array(sensus_df['jk'])
print(f'x_train : {x_train}')
print(f'y_train : {y_train}')


# LabelBinarizer di gunakan untuk mengubah string menjadi angka di y_train
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
print(f'y_train : \n {y_train}')


y_train = y_train.flatten()
print(f'y_train : \n {y_train}')








k = 3
model = KNeighborsClassifier(n_neighbors = k)
model.fit(x_train, y_train)


tinggi_badan = 155
berat_badan = 70
x_new = np.array([tinggi_badan, berat_badan]).reshape(1, -1)
print(f'{x_new}')

# feature harus berformat numpy array dan ukuran 2 dimensi
y_new = model.predict(x_new)
print(y_new)
print(lb.inverse_transform(y_new))








fig, ax = plt.subplots()
for jk, d in sensus_df.groupby('jk'):
	ax.scatter(d['tinggi'], d['berat'], label = jk)


plt.scatter(tinggi_badan, berat_badan, marker = 's', color = 'red', label = 'misterius')
 # marker -> bentuk

plt.legend(loc = 'upper left')  # loc -> mengatur posisi di atas kiri
plt.title('Sebaran Data tinggi badan, berat badan dan jenis jelamin')
plt.xlabel('tinggi (cm)')
plt.ylabel('berat (kg)')
plt.grid(True)
plt.show()







# Kalkulasi Distance (Euclidean Distance)
misterius = np.array([tinggi_badan, berat_badan])
print(misterius)
print(x_train)

data_jarak = [euclidean(misterius, d) for d in x_train]
print(data_jarak)


sensus_df['jarak'] = data_jarak
print(sensus_df.sort_values(['jarak']))  # urutkan dari nilai terkecil ke terbesar di kolom jarak
print(sensus_df)





# Evaluasi permorfa
x_test = np.array([[168, 65], [180, 96], [160, 52], [169, 67]])
y_test = lb.transform(np.array(['pria', 'pria', 'wanita', 'wanita'])).flatten()

print(x_test)
print(y_test)


y_pred = model.predict(x_test)
print(y_pred)













# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy adalah {acc}')


# Precission
prec = precision_score(y_test, y_pred)
print(f'Precission : {prec}')


# Recall
rec = recall_score(y_test, y_pred)
print(f'Recall : {rec}')

# F1
f1 = f1_score(y_test, y_pred)
print(f'F1 : {f1}')

# Classification Report
cls_report = classification_report(y_test, y_pred)
print(f'Classification Report :\n {cls_report}')


# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f'MCC : {mcc}')


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""