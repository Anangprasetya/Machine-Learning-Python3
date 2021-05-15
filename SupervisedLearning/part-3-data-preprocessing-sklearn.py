import numpy as np 
from sklearn import preprocessing

sample_data = np.array([
		[2.1, -1.9, 5.5],
		[-1.5, 2.4, 3.5],
		[0.5, -7.9, 5.6],
		[5.9, 2.3, -5.8],
	])
print(sample_data)
print(sample_data.shape)

# === Binarisation ===
preprocessor = preprocessing.Binarizer(threshold = 0.5)  # nilai <= 0.5 akan di nol kan
binarised_data = preprocessor.transform(sample_data)
print(binarised_data)



# === Scaling ===
preprocessor = preprocessing.MinMaxScaler(feature_range = (0, 1))
preprocessor.fit(sample_data)
scaled_data = preprocessor.transform(sample_data)
print(scaled_data)

scaled_data = preprocessor.fit_transform(sample_data)
print(scaled_data)




# === l1 Normalisation: Least Absolute Deviations ===
li_normalised_data = preprocessing.normalize(sample_data, norm = 'l1')
print(li_normalised_data)



# === l2 Normalisation: Least Squares ===
l2_normalised_data = preprocessing.normalize(sample_data, norm = 'l2')
print(l2_normalised_data)


"""
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
))))))))))))))))))))))))))))      ANANG NUR PRASETYA       ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))             2021              ))))))))))))))))))))))))))))
))))))))))))))))))))))))))))    Simple Machine Learning    ))))))))))))))))))))))))))))
)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
"""