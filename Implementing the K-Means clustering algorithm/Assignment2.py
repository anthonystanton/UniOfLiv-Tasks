# Loading the libraries
import pandas as pd
import numpy as np
import scipy
from scipy import spatial
import matplotlib.pyplot as plt
import copy
from collections import Counter
import operator
from scipy import special
from sklearn.metrics import precision_score, recall_score

# Loading the files into the data frames
animals = pd.read_csv("./clustering-data/animals", sep=" ", header=None)
countries = pd.read_csv("./clustering-data/countries", sep= " ", header=None)
fruits = pd.read_csv("./clustering-data/fruits", sep=" ", header=None)
veggies = pd.read_csv("./clustering-data/veggies", sep=" ", header=None)

# Adding the categories from the data frames
animals['label'] = 'animals'
countries['label'] = 'countries'
fruits['label'] = 'fruits'
veggies['label'] = 'veggies'

animals_size = animals.shape[0]
countries_size = countries.shape[0]
fruits_size = fruits.shape[0]
veggies_size = veggies.shape[0]
# Combining all the data provided into a single dataset
all_data = pd.DataFrame(columns=animals.columns)
all_data = all_data.append([animals, countries, fruits, veggies], ignore_index=True)

# Extract the data and labels from the features
labels = all_data['label'].values
x_data = all_data.drop([0, 'label'], axis=1).values

def k_means(k, x_data, distance_measure, normalize_=False,  max_iterations=1000):
    if normalize_:
        # Normalize with l2     
        l2_x_data = x_data / np.linalg.norm(x_data)
        # Randomly initialize the centroids
        l2_centroids = l2_x_data[np.random.randint(l2_x_data.shape[0], size=k)]
        # Initialize zero vectors to store the category of each data point
        # and the distance from each centroid after running the algorithm
        categories = np.zeros(l2_x_data.shape[0], dtype=np.float64)
        _distance = np.zeros([l2_x_data.shape[0], k], dtype=np.float64)
        for iter in range(max_iterations):
            # Assign each points to the closest cluster
            if distance_measure == "euc":
                for pos, centr in enumerate(l2_centroids):
                    _distance[:, pos] = np.linalg.norm(l2_x_data - centr, axis=1)
            elif distance_measure == "manh":
                for pos, centr in enumerate(l2_centroids):
                    _distance[:, pos] = np.sum(np.abs(l2_x_data - centr), axis=1)
            elif distance_measure == "cos":
                for pos, centr in enumerate(l2_centroids):
                    #_distance[:, pos] = spatial.distance.cosine(l2_x_data, centr)
                    _distance[:, pos] = 1 - (np.dot(l2_x_data, centr) / (np.linalg.norm(l2_x_data, axis=1) * np.linalg.norm(centr)))
                
            # Set the current category of each point as its closest centroid
            categories = np.argmin(_distance, axis=1)
            # Find the new centroids by taking the mean value of the assigned points  
            for categ in range(k):
                l2_centroids[categ] = np.mean(l2_x_data[categories == categ], 0)
    else:
        # Randomly initialize the centroids
        centroids = x_data[np.random.randint(x_data.shape[0], size=k)]
        # Initialize zero vectors to store the category of each data point
        # and the distance from each centroid after running the algorithm
        categories = np.zeros(x_data.shape[0], dtype=np.float64)
        _distance = np.zeros([x_data.shape[0], k], dtype=np.float64)
        for iter in range(max_iterations):
            # Assign each points to the closest cluster
            if distance_measure == "euc":
                for pos, centr in enumerate(centroids):
                    _distance[:, pos] = np.linalg.norm(x_data - centr, axis=1)
            elif distance_measure == "manh":
                for pos, centr in enumerate(centroids):
                    _distance[:, pos] = np.sum(np.abs(x_data - centr), axis=1)
            elif distance_measure == "cos":
                for pos, centr in enumerate(centroids):
                    _distance[:, pos] = 1 - (np.dot(x_data, centr) / (np.linalg.norm(x_data, axis=1) * np.linalg.norm(centr)))
            # Set the current category of each point as its closest centroid
            categories = np.argmin(_distance, axis=1)
            # Find the new centroids by taking the mean value of the assigned points  
            for categ in range(k):
                centroids[categ] = np.mean(x_data[categories == categ], 0)
    return categories

def evaluate(k, categories):
    categories = list(categories)
    # Initialize list to store labels that have already been identified
    identified_categories = []
    # Initialize list to store the number of correctly identified entries per category
    correctly_id_per_category = []
    # Initialize list to store the totals per category
    totals_per_category = [animals_size, countries_size, 
                                    fruits_size, veggies_size]
    # Split the list of all categories to the four expected chunks (based on the size of the original
    # individual data)
    expected_animal_categ_loc = categories[:animals_size]
    current_index = animals_size
    expected_countries_categ_loc = categories[current_index:(current_index + countries_size)]
    current_index += countries_size
    expected_fruits_categ_loc = categories[current_index:(current_index + fruits_size)]
    current_index += fruits_size
    expected_veggies_categ_loc = categories[current_index:]
    
    # Put the lists above in a list
    split_categories = [expected_animal_categ_loc, expected_countries_categ_loc, 
                        expected_fruits_categ_loc, expected_veggies_categ_loc]
    # Initialize list to store classes indexes after identification
    numerical_label = []
    for predicted_category in split_categories:
        # Count number of occurrences of each category in list
        category_count = Counter(predicted_category)
        # Get category label with maximum count and the maximum count itself
        max_in_count = max(category_count.items(), key=operator.itemgetter(1))
        initial_max = max_in_count[0]
        # If the label with max count has already been tied to another class, 
        # Get next label with maximum count
        while max_in_count[0] in identified_categories:
            try:
                # Delete the entry from the dict
                del category_count[max_in_count[0]]
                # Get next maximum value
                max_in_count = max(category_count.items(), key=operator.itemgetter(1))
            except:
                max_in_count = ("False", 0)
        # Append new label to identified categories' list  
        if max_in_count[0] != "False":
            identified_categories.append(max_in_count[0])
            numerical_label += [max_in_count[0]] * len(predicted_category)
        else:
            numerical_label += [initial_max] * len(predicted_category)
        correctly_id_per_category.append(max_in_count[1])
        
    precision = precision_score(numerical_label, categories, average='macro')
    recall = recall_score(numerical_label, categories, average='macro')
    f_score = 2*(recall * precision) / (recall + precision)
    return precision, recall, f_score
    

def question1_2():
	precision_values = [1]
	recall_values = [1]
	f_score_values = [1]
	k_values = list(range(1, 11))
	for k in k_values[1:]:
		categories = k_means(k, x_data, "euc", normalize_=False,  max_iterations=1000)
		results = evaluate(k, categories)
		precision_values.append(results[0])
		recall_values.append(results[1])
		f_score_values.append(results[2])
	plt.plot(k_values, precision_values, marker='+', color='skyblue', linewidth=2)
	plt.plot(k_values, recall_values, marker='x', color='purple', linewidth=2)
	plt.plot(k_values, f_score_values, marker='o', color='red', linewidth=3)
	plt.xlabel("k")
	plt.ylabel("Evaluation parameters")
	plt.legend(['Precision', 'Recall', 'F Score'], loc='upper right')
	plt.title("K-means clustering algorithm with Euclidean distance.")
	plt.show()
	# plt.clf()

def question3():
	precision_values = [1]
	recall_values = [1]
	f_score_values = [1]
	k_values = list(range(1, 11))
	for k in k_values[1:]:
		categories = k_means(k, x_data, "euc", normalize_=True,  max_iterations=1000)
		results = evaluate(k, categories)
		precision_values.append(results[0])
		recall_values.append(results[1])
		f_score_values.append(results[2])
	plt.plot(k_values, precision_values, marker='+', color='skyblue', linewidth=2)
	plt.plot(k_values, recall_values, marker='x', color='purple', linewidth=2)
	plt.plot(k_values, f_score_values, marker='o', color='red', linewidth=3)
	plt.xlabel("k")
	plt.ylabel("Evaluation parameters")
	plt.legend(['Precision', 'Recall', 'F Score'], loc='upper right')
	plt.title("K-means clustering algorithm with Euclidean distance and L2 normalization.")
	plt.show()

def question4():
	precision_values = [1]
	recall_values = [1]
	f_score_values = [1]
	k_values = list(range(1, 11))
	for k in k_values[1:]:
		categories = k_means(k, x_data, "manh", normalize_=False,  max_iterations=1000)
		results = evaluate(k, categories)
		precision_values.append(results[0])
		recall_values.append(results[1])
		f_score_values.append(results[2])
	plt.plot(k_values, precision_values, marker='+', color='skyblue', linewidth=2)
	plt.plot(k_values, recall_values, marker='x', color='purple', linewidth=2)
	plt.plot(k_values, f_score_values, marker='o', color='red', linewidth=3)
	plt.xlabel("k")
	plt.ylabel("Evaluation parameters")
	plt.legend(['Precision', 'Recall', 'F Score'], loc='upper right')
	plt.title("K-means clustering algorithm with Manhattan distance.")
	plt.show()

def question5():
	precision_values = [1]
	recall_values = [1]
	f_score_values = [1]
	k_values = list(range(1, 11))
	for k in k_values[1:]:
		categories = k_means(k, x_data, "manh", normalize_=True,  max_iterations=1000)
		results = evaluate(k, categories)
		precision_values.append(results[0])
		recall_values.append(results[1])
		f_score_values.append(results[2])
	plt.plot(k_values, precision_values, marker='+', color='skyblue', linewidth=2)
	plt.plot(k_values, recall_values, marker='x', color='purple', linewidth=2)
	plt.plot(k_values, f_score_values, marker='o', color='red', linewidth=3)
	plt.xlabel("k")
	plt.ylabel("Evaluation parameters")
	plt.legend(['Precision', 'Recall', 'F Score'], loc='upper right')
	plt.title("K-means clustering algorithm with Manhattan distance with L2 normalizatin.")
	plt.show()

def question6():
	precision_values = [1]
	recall_values = [1]
	f_score_values = [1]
	k_values = list(range(1, 11))
	for k in k_values[1:]:
		categories = k_means(k, x_data, "cos", normalize_=False,  max_iterations=1000)
		results = evaluate(k, categories)
		precision_values.append(results[0])
		recall_values.append(results[1])
		f_score_values.append(results[2])
	plt.plot(k_values, precision_values, marker='+', color='skyblue', linewidth=2)
	plt.plot(k_values, recall_values, marker='x', color='purple', linewidth=2)
	plt.plot(k_values, f_score_values, marker='o', color='red', linewidth=3)
	plt.xlabel("k")
	plt.ylabel("Evaluation parameters")
	plt.legend(['Precision', 'Recall', 'F Score'], loc='upper right')
	plt.title("K-means clustering algorithm with Cosine distance.")
	plt.show()

question1_2()
plt.clf()
question3()
plt.clf()
question4()
plt.clf()
question5()
plt.clf()
question6()
plt.clf()