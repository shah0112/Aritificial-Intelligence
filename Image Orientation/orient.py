'''Adaboost Algorithm:

The algorithm given in Freund & Schapire was used for implementing Adaboost. 

Step 1: 6 Classifiers were built for identifying:
	Orientation 0 vs Orientation 90 
	Orientation 0 vs Orientation 180 
	Orientation 0 vs Orientation 270 
	Orientation 90 vs Orientation 180 
	Orientation 90 vs Orientation 270 
	Orientation 180 vs Orientation 270 
Step 2: For each classifier, 20 stumps were built. Each stump compares the value of two pixels. 192 x 191 stumps 
were created to compare if pixel_i ≥ pixel_j
Step 3: To choose the stumps, the weighted error was calculated for each of the stumps and the stump with least 
weighted error was chosen. The weights of the images are initialized based on positive and negative sample distribution
Step 4: After choosing the first stump, prediction is done on the images and correctly classified images are down-weighted. 
The weights are then normalized, and the subsequent stumps are created in a similar manner
Step 5: The pixels compared by the stumps and the alpha value corresponding to the stumps are stored
Step 6: Other 5 classifiers are built in a similar manner, taking the same number of images as input and the same 
number of stumps
Step 7: The data label is predicted by each stump of the classifier. This predicted vector is multiplied by the 
alpha value of the corresponding stump. The vectors produced by the 20 stumps are added to generate a final vector for the classifier
Step 8: Each of the 6 classifiers predictions are swapped to generate predictions for the opposite classes 
Step 9: The final prediction is done by comparing the 12 vectors produced in Step 7. The class corresponding 
to maximum value is chosen as the predicted class
'''

# Importing the required packages
import numpy as np
from operator import itemgetter
import json

(variable, file_name, model_file, model) = sys.argv[1:]

# Produce 1/0 label based on the classes for which the classifier is built
def label_estimator(label_data, label_value):
    return np.where(label_data == label_value, 1, 0) 


# Split the data based on classes
def train_data_split(data, label, class1, class2):
    row_idx = (np.argwhere((label == class1) | (label == class2))).ravel()
    split_data = data[row_idx,:]
    split_label = label_estimator(label[row_idx], class1)    
    return split_data, split_label


# Epsilon value: To calculate weighted error
def epsilon(sample_weight, hypothesis_pred, label_train):
    return np.sum(sample_weight*abs(hypothesis_pred-label_train))

# Re-weighting samples based on error from previous prediction
def sample_wt_estimator(stump_no, prev_weight, label_train, prev_hypothesis, N):
    if stump_no == 0:
        return np.where(label_train == 1, 1.0/(2*np.sum(label_train)), 1.0/(2*(N-np.sum(label_train))))
    else:
        epsilon_t = epsilon(prev_weight, prev_hypothesis, label_train)
        beta = epsilon_t/(1 - epsilon_t)
        weights = np.where(prev_hypothesis == label_train, prev_weight*beta, prev_weight)
        return weights/ np.sum(weights)

# Creating 20 stumps classifier based on the error:
def stump_creator(label_train, N, truth_dict):
    sample_weight = np.zeros(N)
    stumps = {}
    stump_keys = []
    tot_alpha = 0
    best_pred = np.zeros([N])
    for stump_no in range(20):
        sample_weight = sample_wt_estimator(stump_no, sample_weight, label_train, best_pred, N)
        error = 10000
        for key, pred in truth_dict.iteritems():
            if key not in stump_keys:
                [i,j] = key
                e = epsilon(sample_weight, pred, label_train)
                if e < error:
                    error = e
                    best_pred = pred
                    best_i = i
                    best_j = j
                    epsilon_val = epsilon(sample_weight, best_pred, label_train)
                    beta = epsilon_val/(1-epsilon_val)
                    alpha = np.log(1/beta)
        tot_alpha += alpha
        stumps[(best_i,best_j)] = alpha
        stump_keys = [key for key, value in stumps.iteritems()]
    return stumps

# Aggregating the steps required for training the model
def main(label_train,train_data):
    feature = train_data.transpose()
    N = len(label_train)  # N is the sample size
    K = len(feature) # K is the number of stumps
    truth_dict = {}
    for i in range(K):
        for j in range(K):
            if (i != j):
                truth_dict[(i,j)] = np.where(feature[i] >= feature[j], 1, 0)
    alpha_pred = stump_creator(label_train, N, truth_dict)
    return alpha_pred

# Predicts 1/-1 and multiplies by alpha for each classifier  
def prediction(dictionary, train_data):
    N = len(train_data)
    pred_output = np.zeros([N])
    for pair, alpha in dictionary.iteritems():
        i, j = pair
        alpha = (np.sum(alpha))
        list1 = train_data[:, i]
        list2 = train_data[:, j]
        output = np.where( list1 >= list2, alpha, -1*alpha )
        pred_output = np.add(pred_output,output)
    return pred_output

def dict_correcter(old_dict):
    new_dict = {}
    for key, value in old_dict.iteritems():
        a = key.encode('utf-8')
        new_key = tuple(map(lambda x: int(filter(str.isdigit,x)) ,a.split(',')))
        new_dict[(new_key)] = value
    return (new_dict)

def train_adaboost(file_name, model_file):
    # Reading the features and labels into separate numpy arrays 
    train_data_full = np.loadtxt(file_name, delimiter = ' ', usecols = range(2,194), dtype = int)
    train_label_full = np.loadtxt(file_name, delimiter = ' ', usecols = 1, dtype = str)
    
    # Splitting the input data and labels for each of the 6 classifiers 
    train_090, label_090 = train_data_split(train_data_full, train_label_full, '0', '90')
    train_0180, label_0180 = train_data_split(train_data_full, train_label_full, '0', '180')
    train_0270, label_0270 = train_data_split(train_data_full, train_label_full, '0', '270')
    train_90180, label_90180 = train_data_split(train_data_full, train_label_full, '90', '180')
    train_90270, label_90270 = train_data_split(train_data_full, train_label_full, '90', '270')
    train_180270, label_180270 = train_data_split(train_data_full, train_label_full, '180', '270')
    
    # Training 6 classifiers
    orient_0_90 = main(label_090, train_090)
    orient_0_180 = main(label_0180, train_0180)
    orient_0_270 = main(label_0270, train_0270)
    orient_90_180 = main(label_90180, train_90180)
    orient_90_270 = main(label_90270, train_90270)
    orient_180_270 = main(label_180270, train_180270)
    
    # Creating 6 other classifiers based on the previous 6:
    orientation_90_0    = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_90.iteritems()}
    orientation_180_0   = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_180.iteritems()}
    orientation_270_0   = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_270.iteritems()}
    orientation_180_90  = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_90_180.iteritems()}
    orientation_270_90  = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_90_270.iteritems()}
    orientation_270_180 = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_180_270.iteritems()}
    orientation_0_90    = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_90.iteritems()}
    orientation_0_180   = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_180.iteritems()}
    orientation_0_270   = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_0_270.iteritems()}
    orientation_90_180  = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_90_180.iteritems()}
    orientation_90_270  = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_90_270.iteritems()}
    orientation_180_270 = {('('+ str(key[1])+','+str(key[0])+')'): value for key , value in orient_180_270.iteritems()}
    
    train_final_dict = {
    'orientation_0_90' : orientation_0_90,
    'orientation_0_180' : orientation_0_180,
    'orientation_0_270' : orientation_0_270,
    'orientation_90_0' : orientation_90_0,
    'orientation_90_180' : orientation_90_180,
    'orientation_90_270' : orientation_90_270,
    'orientation_180_0' : orientation_180_0,
    'orientation_180_90' : orientation_180_90,
    'orientation_180_270' : orientation_180_270,
    'orientation_270_0' : orientation_270_0,
    'orientation_270_90' : orientation_270_90,
    'orientation_270_180' : orientation_270_180,
    }
    # Writing model to output file:
    with open(model_file, 'w') as file:
        file.write(json.dumps(train_final_dict))

def test_adaboost(file_name, model_file):
    # Reading the features and labels into separate numpy arrays
    test_image_full = np.loadtxt(file_name, delimiter = ' ', usecols = 0, dtype = int)
    test_data_full = np.loadtxt(file_name, delimiter = ' ', usecols = range(2,194), dtype = int)
    test_label_full = np.loadtxt(file_name, delimiter = ' ', usecols = 1, dtype = str)
    model = json.load(open(model_file))
    overall_dict = eval('model')
    
    # Converting into separate dictionaries in the right format
    orientation_0_90 = dict_correcter(overall_dict['orientation_0_90'])
    orientation_0_180 = dict_correcter(overall_dict['orientation_0_180'])
    orientation_0_270 = dict_correcter(overall_dict['orientation_0_270'])
    orientation_90_0 = dict_correcter(overall_dict['orientation_90_0'])
    orientation_90_180 = dict_correcter(overall_dict['orientation_90_180'])
    orientation_90_270 = dict_correcter(overall_dict['orientation_90_270'])
    orientation_180_0 = dict_correcter(overall_dict['orientation_180_0'])
    orientation_180_90 = dict_correcter(overall_dict['orientation_180_90'])
    orientation_180_270 = dict_correcter(overall_dict['orientation_180_270'])
    orientation_270_0 = dict_correcter(overall_dict['orientation_270_0'])
    orientation_270_90 = dict_correcter(overall_dict['orientation_270_90'])
    orientation_270_180 = dict_correcter(overall_dict['orientation_270_180'])

    # Importing the model file and inserting stumps into dictionaries:
    ori090_test_pred = prediction(orientation_0_90, test_data_full)
    ori0180_test_pred = prediction(orientation_0_180, test_data_full)
    ori0270_test_pred = prediction(orientation_0_270, test_data_full)
    ori900_test_pred = prediction(orientation_90_0, test_data_full)
    ori90180_test_pred = prediction(orientation_90_180, test_data_full)
    ori90270_test_pred = prediction(orientation_90_270, test_data_full)
    ori1800_test_pred = prediction(orientation_180_0, test_data_full)
    ori18090_test_pred = prediction(orientation_180_90, test_data_full)
    ori180270_test_pred = prediction(orientation_180_270, test_data_full)
    ori2700_test_pred = prediction(orientation_270_0, test_data_full)
    ori27090_test_pred = prediction(orientation_270_90, test_data_full)
    ori270180_test_pred = prediction(orientation_270_180, test_data_full)
    
    ori0_test_pred   = (ori090_test_pred  + ori0180_test_pred  +   ori0270_test_pred)/3
    ori90_test_pred  = (ori900_test_pred  + ori90180_test_pred +  ori90270_test_pred)/3
    ori180_test_pred = (ori1800_test_pred + ori18090_test_pred + ori180270_test_pred)/3
    ori270_test_pred = (ori2700_test_pred + ori27090_test_pred + ori270180_test_pred)/3
    
    final_test_pred = []
    for i in range(len(test_data_full)):
        pred_list = [ori0_test_pred[i], ori90_test_pred[i], ori180_test_pred[i], ori270_test_pred[i]]
        max_index = pred_list.index(max(pred_list))
        if max_index == 0:
            label = '0'
        elif max_index == 1:
            label = '90'
        elif max_index == 2:
            label = '180'
        else:
            label = '270'
        final_test_pred.append(label)
    print("Accuracy: ",np.sum(np.where(final_test_pred == test_label_full, 1, 0))*100.0/len(test_label_full),"%")
    final_output = np.concatenate(test_image_full, final_test_pred.astype(int), axis = 1)
    np.savetxt('output.txt',final_output)