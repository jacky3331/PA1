import numpy as np

def extract_data(file):
    data = []
    with open(file) as f:
        for line in f.readlines():
            data.append([int(i) for i in line.strip().split(" ")])
            
    return data

def extract_double_data(file):
    data = []
    with open(file) as f:
        for line in f.readlines():
            data.append([float(i) for i in line.strip().split(" ")])
            
    return data

def find_distance(vector1, vector2):
    vector = np.square(np.subtract(vector1, vector2))
    distance = np.sqrt(np.sum(vector))
    
    return distance

def find_neighbors(train_set, input_vector, k):
    distance_set = []
    neighbors_set = []
    for train_vector in train_set:
        distance = find_distance(train_vector[:784], input_vector)
        distance_set.append((train_vector, distance))
    distance_set.sort(key=lambda tup: tup[1])
    
    for i in range(k):
        neighbors_set.append(distance_set[i][0])
        
    return neighbors_set

train_set = extract_data("pa1train.txt")
val_set = extract_data("pa1validate.txt")
test_set = extract_data("pa1test.txt")
proj_matrix = extract_double_data("projection.txt")

# This chunk of code evaluates the error of the KNN algorithm (part 1).

output_labels = []
actual_labels = []
error_count = 0

for j in range(len(test_set)):
    neighbors = find_neighbors(train_set, test_set[j][:784], 1)
    output = [int(i[-1]) for i in neighbors]
    label = max(set(output), key=output.count)

    if label != test_set[j][-1]:
        error_count += 1

print(error_count/len(test_set))


# This chunk of code utilizes projections to evaluate the error of the KNN algorithm (part 2).

train_label = [i[-1] for i in train_set]
val_label = [i[-1] for i in val_set]
test_label = [i[-1] for i in test_set]

train_set = np.column_stack((np.matmul(np.dot([i[:784] for i in train_set], proj_matrix), np.transpose(proj_matrix))/np.linalg.norm(proj_matrix), train_label))
val_set = np.column_stack((np.matmul(np.dot([i[:784] for i in val_set], proj_matrix), np.transpose(proj_matrix))/np.linalg.norm(proj_matrix), val_label))
test_set = np.column_stack((np.matmul(np.dot([i[:784] for i in test_set], proj_matrix), np.transpose(proj_matrix))/np.linalg.norm(proj_matrix), test_label))

output_labels = []
actual_labels = []
error_count = 0

for j in range(len(test_set)):
    neighbors = find_neighbors(train_set, test_set[j][:784], 9)
    output = [int(i[-1]) for i in neighbors]
    label = int(max(set(output), key=output.count))

    if label != test_set[j][-1]:
        error_count += 1

print(error_count/len(test_set))
