import numpy as np
import csv
from sklearn.datasets.base import Bunch
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

FEATURES_NUM = 13
SAMPLES_NUM = 303

TREE_NUM = 5
TREE_DATA_NUM = 150


def load_dataset():
	with open('heart.csv') as csv_file:
		data_file = csv.reader(csv_file)
		temp = next(data_file)
		n_samples = SAMPLES_NUM #number of data rows, don't count header
		n_features = FEATURES_NUM #number of columns for features, don't count target column
		feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] #adjust accordingly
		target_names = ['target'] #adjust accordingly
		data = np.empty((n_samples, n_features))
		target = np.empty((n_samples,), dtype=np.int)

		for i, sample in enumerate(data_file):
			data[i] = np.asarray(sample[:-1], dtype=np.float64)
			target[i] = np.asarray(sample[-1], dtype=np.int)

	return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)

def get_tree(data, target):
	dec_tree = tree.DecisionTreeClassifier()
	dec_tree = dec_tree.fit(data, target)
	return dec_tree

def get_predict_target(data, dec_tree):
	predict_target = dec_tree.predict(data)
	return predict_target

def sample_dataset(dataset, number):
	n_samples = number 
	n_features = 13 
	feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] 
	target_names = ['target'] 
	#data = np.empty((n_samples, n_features))
	#target = np.empty((n_samples,), dtype=np.int)
	data = dataset.data.copy()
	target = dataset.target.copy()

	for i in range(len(dataset.data) - number):
		random_number = random.randrange(0, len(data))
		data = np.delete(data, random_number, 0)
		target = np.delete(target, random_number, 0)

	return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)

def sample(data, target, number):
	if number > len(data):
		print("Error: Number of sampling", number, " is greater than number of all train data", len(data), "!")
		return None
	n_samples = number 
	n_features = 13 
	feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] 
	target_names = ['target'] 
	sample_data = np.empty((n_samples, n_features))
	sample_target = np.empty((n_samples,), dtype=np.int)
	for i in range(number):
		random_number = random.randrange(0, len(data))
		sample_data[i] = data[random_number]
		sample_target[i] = target[random_number]

	return Bunch(data=data, target=target, feature_names = feature_names, target_names = target_names)
	
def calculate_end_target(predict_targets):
	end_target = []
	for i in range(len(predict_targets[0])):
		target_numbers = {}
		max_number = 0
		for j in range(len(predict_targets)):
			if predict_targets[j][i] in target_numbers:
				target_number = target_numbers.get(predict_targets[j][i])
			else:
				target_number = 1
			if target_number > max_number:
				max_number = target_number
				mod_target = predict_targets[j][i]
			target_numbers.update({predict_targets[j][i]:target_number})
		end_target += [mod_target]
	return end_target

def calculate_accuracy(predict_target, true_target):
	trues = 0
	for i in range(len(true_target)):
		if predict_target[i] == true_target[i]:
			trues += 1
	return trues / len(true_target)

def mask_data(data, mask_array):
	mask = np.array(mask_array)
	return data[:, mask]

def generate_mask_array(features_num, removed_feature_index):
	mask_array = []
	for i in range(features_num):
		mask_array += [True]
	mask_array[removed_feature_index] = False
	return mask_array

def gen_mask_array_5(features_num):
	mask_array = []
	for i in range(features_num):
		mask_array += [False]
	selected = 0
	while True:
		random_number = random.randrange(0, features_num)
		if mask_array[random_number] == False:
			mask_array[random_number] = True
			selected += 1
		if selected == 5:
			break
	return mask_array

dataset = load_dataset()
train_data, test_data, train_target, test_target = train_test_split(dataset.data, dataset.target, test_size = 0.2)
dec_tree = get_tree(train_data, train_target)

# Question 1
print("Q1: ", dec_tree.score(test_data, test_target))

# Question 2 part 1
sample_datasets = []
for i in range(TREE_NUM):
	sample_datasets += [sample(train_data, train_target, TREE_DATA_NUM)]

# Question 2 part 2
dec_trees = []
for sample_dataset in sample_datasets:
	dec_trees += [get_tree(sample_dataset.data, sample_dataset.target)]

predict_targets = []
for dec_tree in dec_trees:
	predict_targets += [get_predict_target(test_data, dec_tree)]

end_target = calculate_end_target(predict_targets)
print("Q2-2: ", calculate_accuracy(end_target, test_target))

# Question 2 part 3
for i in range(FEATURES_NUM):
	mask_array = generate_mask_array(FEATURES_NUM, i)
	masked_data = mask_data(train_data, mask_array)
	dec_tree = get_tree(masked_data, train_target)
	masked_test_data = mask_data(test_data, mask_array)
	print("Q2-3 - removing feature -", dataset.feature_names[i], ":",  dec_tree.score(masked_test_data, test_target))

# Question 2 part 4
mask_array = gen_mask_array_5(FEATURES_NUM)
masked_data = mask_data(train_data, mask_array)
dec_tree = get_tree(masked_data, train_target)
masked_test_data = mask_data(test_data, mask_array)
print("Q2-4:", dec_tree.score(masked_test_data, test_target))

# Question 2 part 5
sample_datasets = []
for i in range(TREE_NUM):
	sample_datasets += [sample(train_data, train_target, TREE_DATA_NUM)]

mask_arrays = []
masked_datas = []
masked_test_datas = []
for i in range(TREE_NUM):
	mask_arrays += [gen_mask_array_5(FEATURES_NUM)]
	masked_datas += [mask_data(sample_datasets[i].data, mask_arrays[i])]
	masked_test_datas += [mask_data(test_data, mask_arrays[i])]

dec_trees = []
predict_targets = []
for i in range(TREE_NUM):
	dec_trees += [get_tree(masked_datas[i], sample_datasets[i].target)]
	predict_targets += [get_predict_target(masked_test_datas[i], dec_trees[i])]

end_target = calculate_end_target(predict_targets)
print("Q2-5: ", calculate_accuracy(end_target, test_target))






