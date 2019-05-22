import UtilityFunctions
import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as k_neigh
from sklearn.ensemble import VotingClassifier as voter
from sklearn.neural_network import MLPClassifier as nn
import pandas as pd
import time
import sys
import warnings

# to remove annoying 'future' and 'soon to be deprecated' warnings
#from sklearn.neural_network.tests.test_mlp import test_n_iter_no_change

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# make it easier to read output
pd.set_option('display.max_columns', 30)

# amount of variation for PCA
global PCA_THRESHOLD
PCA_THRESHOLD = 0.999

# dictionaries for classification
key_binary = {0: 'book', 1: 'plastic case'}
key_multi = {0: 'air', 1: 'book', 2: 'hand', 3: 'knife', 4: 'plastic case'}


def initial_training_of_models(Xfile, Yfile):
    """Performs the initial training on all models. Each model calls upon its own Grid search function"""

    # first obtain and process the training sets
    x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val, x_all, y_all = UtilityFunctions.read_CSV_data(
        Xfile, Yfile)

    # this will help us record the results for ensemble voting
    df_val_predictions = y_val.copy()
    df_val_predictions.reset_index(inplace=True)

    # extract the PCAs from X and plot the correlation heat maps
    x_train_processed, pca = process_data(x_train)
    UtilityFunctions.plot_correlation_heatmap(pd.DataFrame(x_train_processed),
                                              "Correlation heatmap of PCA Components \n(individual labels not shown)",
                                              0.1)
    x_val_processed = process_data_existingPCA(x_val, pca)

    # print the number of PCA features
    num_pca_features = len(x_train_processed[0])
    print("Number of PCA features", num_pca_features)

    # train logisitc regression model
    logReg = logisitic_regression_initial_benchmark(df_val_predictions, x_train_processed, x_val_processed, y_train,
                                                    y_val)

    # perform grid search on Grad Boost Trees
    param_grid_GradBoost = initialise_param_grid_GradientBoosted(num_pca_features)
    grid_search_gradBoost = GradientBoostedTreesGridSearch(x_train_processed, y_train, param_grid_GradBoost)
    UtilityFunctions.model_results(grid_search_gradBoost, x_val_processed, y_val, df_val_predictions, 'GrBoost')

    # perform grid search on KNN
    param_grid_kNearest = [{'n_neighbors': [i for i in range(1, 10)]}]
    grid_search_kNearest = KNearestNeigboursGridSearch(x_train_processed, y_train, param_grid_kNearest)
    UtilityFunctions.model_results(grid_search_kNearest, x_val_processed, y_val, df_val_predictions, 'K-Nearest')

    # perform grid search on NN
    param_grid_nn = initialise_param_grid_NeuralNet()
    grid_search_NN = NeuralNetworkGridSearch(x_train_processed, y_train, param_grid_nn)
    UtilityFunctions.model_results(grid_search_NN, x_val_processed, y_val, df_val_predictions, 'NN')

    # print predictions
    print("\n\n***** Predictions from initial models on validation set")
    print(df_val_predictions)

    return df_val_predictions, x_train_processed, pca, x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val, x_all, y_all, grid_search_gradBoost, grid_search_kNearest, grid_search_NN, logReg;


def process_data(x_values):
    """Scale data using standard scaler and apply PCA transform"""

    x_values_stand = sklearn.preprocessing.StandardScaler().fit_transform(x_values)
    pca = PCA(n_components=PCA_THRESHOLD)
    x_values_stand_pca = pca.fit_transform(x_values_stand)

    # print distribution of PCA variation
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print("\n% of variation each PCA captures:")
    print(cumsum)
    return x_values_stand_pca, pca;


def process_data_existingPCA(x_values, pca):
    """Scale data using standard scaler and apply existing PCA transform"""

    x_values_stand = sklearn.preprocessing.StandardScaler().fit_transform(x_values)
    x_values_stand_pca = pca.transform(x_values_stand)
    return x_values_stand_pca


def logisitic_regression_initial_benchmark(df_val_predictions, x_train_processed, x_val_processed, y_train, y_val):
    """Performs regularised logistic regression"""

    logReg = sklearn.linear_model.LogisticRegression()
    logReg.fit(x_train_processed, y_train)
    print("\n***** Outcomes for logistic regression")
    print("\nLog Regression training set accuracy", logReg.score(x_train_processed, y_train))
    print("\nLog Regression validation set accuracy", logReg.score(x_train_processed, y_train))
    UtilityFunctions.model_results(logReg, x_val_processed, y_val, df_val_predictions, 'Log-Reg')
    return logReg


def GradientBoostedTreesGridSearch(x_train_processed, y_train, param_grid):
    """A function that performs gridsearch and returns the results for gradient boosted trees"""

    grad_booster = sklearn.ensemble.GradientBoostingClassifier(random_state=30)
    grid_search_gradBoost = sk.model_selection.GridSearchCV(grad_booster, param_grid, cv=3, scoring='accuracy')#,
                                                            #n_jobs=-1)
    grid_search_gradBoost.fit(x_train_processed, y_train.values.ravel())

    # print output of best model
    print("\n***** Grid search outcomes for gradient boosting")
    print("Training score: ", grid_search_gradBoost.best_score_)
    print("Best hyper-parameters: ", grid_search_gradBoost.best_params_)
    return grid_search_gradBoost


def KNearestNeigboursGridSearch(x_train_processed, y_train, param_grid):
    """A function that performs gridsearch and returns the results for K-Nearest Neighbours"""

    k_nearest = k_neigh(n_jobs=-1)
    grid_search_k_near = sk.model_selection.GridSearchCV(k_nearest, param_grid, cv=3, scoring='accuracy')
    grid_search_k_near.fit(x_train_processed, y_train.values.ravel())

    # print output of best model
    print("\n***** Grid search outcomes for K-nearest")
    print("Training score: ", grid_search_k_near.best_score_)
    print("Best hyper-parameters: ", grid_search_k_near.best_params_)
    return grid_search_k_near


def NeuralNetworkGridSearch(x_train_processed, y_train, param_grid):
    """A function that performs gridsearch and returns the results for neural networks"""

    neural_network = nn(activation='logistic', random_state=40, max_iter=1500, momentum=0, solver='sgd',
                        early_stopping=True,  validation_fraction=0.2)
    grid_search_NN = sk.model_selection.GridSearchCV(neural_network, param_grid, cv=3, scoring='accuracy')
    grid_search_NN.fit(x_train_processed, y_train.values.ravel())

    # print output of best model
    print("\n***** Grid search outcomes for neural network")
    print("Training score: ", grid_search_NN.best_score_)
    print("Best hyper-parameters: ", grid_search_NN.best_params_)
    return grid_search_NN


def initialise_param_grid_GradientBoosted(num_pca_features):
    """Returns the initial hyper-parameters for grid search for GBTs"""

    learning_rate_grad_boost = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_depth = [2 ** i for i in range(1, 6)]

    # don't want features exceeding PCA factors
    max_features = [(2 * i) for i in range(1, 15, 2) if (2 * i) <= num_pca_features]
    n_estimators = [10 * i for i in range(1, 11)]

    param_grid = [{'max_depth': max_depth, 'learning_rate': learning_rate_grad_boost, 'max_features': max_features,
                   'n_estimators': n_estimators}]

    return param_grid


def initialise_param_grid_NeuralNet():
    """Returns the initial hyper-parameters for grid search for Neural Nets"""

    units_per_layer = [(i) for i in range(10, 111, 20)]
    units_per_layer += [(i, j) for i in range(10, 111, 20) for j in range(10, 111, 20)]
    learning_rate_init = [0.01, 0.1, 1]
    alpha = [0.001, 0.01, 0.1, 1]

    # print topologies to check
    print("\nNeural net topologies to test")
    print(units_per_layer)

    param_grid = [{'hidden_layer_sizes': units_per_layer, 'alpha': alpha, 'learning_rate_init': learning_rate_init}]

    return param_grid


def fine_tune_binary_models(x_train, y_train, x_val, y_val, df_val_predictions, pca, params_gradBoost_intial,
                            params_NN_initial):

    """Calls upon various grid-search functions to fine-tune the previous model iterations of Gradient boosted trees
    and the neural network"""

    x_train_processed = process_data_existingPCA(x_train, pca)
    x_val_processed = process_data_existingPCA(x_val, pca)

    # obtain the fine-tuned parameters for GB Trees and NN's
    param_grid_GradBoost, param_grid_nn = calulate_fine_tuned_grid_parameters(params_NN_initial,
                                                                              params_gradBoost_intial)
    # perform fine-tuned grid search for GB Trees
    grid_search_gradBoost = GradientBoostedTreesGridSearch(x_train_processed, y_train, param_grid_GradBoost)
    UtilityFunctions.model_results(grid_search_gradBoost, x_val_processed, y_val, df_val_predictions,
                                   'GrBoost - finetuned')

    # perform fine-tuned grid search for Neural Nets
    grid_search_NN = NeuralNetworkGridSearch(x_train_processed, y_train, param_grid_nn)
    UtilityFunctions.model_results(grid_search_NN, x_val_processed, y_val, df_val_predictions, 'NN - finetuned')

    # print validation set results
    print("\n\n***** Predictions from fine-tuned models on validation set")
    print(df_val_predictions)

    return grid_search_gradBoost, grid_search_NN;


def calulate_fine_tuned_grid_parameters(params_NN_initial, params_gradBoost_intial):

    """Provides fine-tuned parameter values for GBTs and NN's for grid search"""

    # fine tune Gradboosted trees hyper-parameters
    learning_rate_grad_boost = [params_gradBoost_intial['learning_rate'] + i / 100 for i in range(-5, 6, 1) if
                                0 < (params_gradBoost_intial['learning_rate'] + i / 100)]

    max_depth = [params_gradBoost_intial['max_depth'] + i for i in range(-2, 3, 1) if
                 0 < params_gradBoost_intial['max_depth'] + i]

    max_features = [params_gradBoost_intial['max_features'] + i for i in range(-2, 3, 1) if
                    0 < params_gradBoost_intial['max_features'] + i]

    n_estimators = [params_gradBoost_intial['n_estimators'] + i for i in range(-2, 3, 1) if
                    0 < params_gradBoost_intial['n_estimators'] + i]

    param_grid_GradBoost = [
        {'max_depth': max_depth, 'learning_rate': learning_rate_grad_boost, 'max_features': max_features,
         'n_estimators': n_estimators}]

    # fine tune Neural Network

    # fine tune topology
    hidden_layer = params_NN_initial['hidden_layer_sizes']
    if (isinstance(hidden_layer, int) or len(hidden_layer) == 1):
        units_per_layer = [hidden_layer + i for i in range(-5, 6, 1)]
    else:
        units_L1 = hidden_layer[0]
        units_L2 = hidden_layer[1]
        units_per_layer = [(units_L1 + i, units_L2 + j) for i in range(-10, 11, 2) for j in range(-10, 11, 2)]

    # fine tune learning rate
    learn_rate_initial = params_NN_initial['learning_rate_init']
    learning_rate_init = [learn_rate_initial + i / 10 for i in range(-2, 3, 1) if
                          0 < learn_rate_initial + i / 10]

    # fine tune regularisation
    alpha_initial = params_NN_initial['alpha']
    alpha = [alpha_initial + i * alpha_initial / 10 for i in range(-3, 3, 1) if
             0 < alpha_initial + i * alpha_initial / 10]


    param_grid_nn = [{'hidden_layer_sizes': units_per_layer, 'alpha': alpha, 'learning_rate_init': learning_rate_init}]

    return param_grid_GradBoost, param_grid_nn



###### Main Program Flow Begins Here ###########

print("\n**** USER FYI: This program takes circa 15-20 minutes to completely run **** \n")

print("\n\n----------------- Binary --------------------------------------------------------------------------------\n")
Xfile = 'X.csv'
Yfile = 'y.csv'

# Train initial model
df_val_predictions_binary, x_train_processed, pca, x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val, x_all, y_all, grid_search_gradBoost, grid_search_kNearest, grid_search_NN, log_regress = initial_training_of_models(
    Xfile, Yfile)

# Fine tune the initial Gradient Boosted Tree and Neural Network models
params_gradBoost = grid_search_gradBoost.best_params_
params_NN = grid_search_NN.best_params_
binary_GradBoost_final, binary_NN_final = fine_tune_binary_models(x_train, y_train, x_val, y_val,
                                                                  df_val_predictions_binary, pca,
                                                                  params_gradBoost, params_NN)

# Add models into a voting ensemble
voting_clf_binary = voter(estimators=[('Logistic Regression', log_regress), ('gradBoost', binary_GradBoost_final),
                                      ('K_nearest', grid_search_kNearest),
                                      ('Neural network', binary_NN_final)], voting='hard')
voting_clf_binary.fit(x_train_processed, y_train)
model_name = "Ensemble - Hard Voting"

# print ensemble results on training set
print("\n***** Binary, ensemble (training set)")
UtilityFunctions.model_results(voting_clf_binary, x_train_processed, y_train, df_val_predictions_binary,
                               model_name)

# Test ensemble on validation set
x_val_processed = process_data_existingPCA(x_val, pca)
print("\n***** Binary, ensemble (validation set)")
UtilityFunctions.model_results(voting_clf_binary, x_val_processed, y_val, df_val_predictions_binary,
                               model_name)

print("\n\n***** Predictions from binary ensemble model on validation set")
print(df_val_predictions_binary)

# Test ensemble on test set
print("\n***** Binary, ensemble (test set)")
x_train_val_processed = process_data_existingPCA(x_train_val, pca)
voting_clf_binary.fit(x_train_val_processed, y_train_val)
x_test_processed = process_data_existingPCA(x_test, pca)
UtilityFunctions.model_results(voting_clf_binary, x_test_processed, y_test, df_val_predictions_binary,
                               model_name)

# retrain on all data and make predictions on unseen data
x_all_processed = process_data_existingPCA(x_all, pca)
voting_clf_binary.fit(x_all_processed, y_all)
x_unlabelled = UtilityFunctions.read_unlabelled_CSV_data('XToClassifyBin.csv')
x_unlabelled_processed = process_data_existingPCA(x_unlabelled, pca)
y_pred = voting_clf_binary.predict(x_unlabelled_processed)

print("\n\n----------------- Binary Training Finished --------------------------------------------------------------\n")

# write predictions to file
file1 = open("PredictedClassesBin.csv", "w")

print("\nPredictions for unlabelled binary data\n")
print("Row,Prediction,Category")
file1.write("Row,Prediction,Category\n")
for i in range(0, len(y_pred)):
    string_line = str(i) + "," + str(y_pred[i]) + "," + str(key_binary[y_pred[i]])
    print(string_line)
    file1.write(string_line + "\n")
file1.close()





# begin multi-class training
print("\n\n----------------- Starting Multi-class training ---------------------------------------------------------\n")

XfileMulti = 'XMulti.csv'
YfileMulti = 'yMulti.csv'

# Train initial model
df_val_predictions_multi, x_train_processed, pca, x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val, x_all, y_all, grid_search_gradBoost, grid_search_kNearest, grid_search_NN, log_regress = initial_training_of_models(
    XfileMulti, YfileMulti)

# Fine tune the initial Gradient Boosted Tree and Neural Network models
params_gradBoost = grid_search_gradBoost.best_params_
params_NN = grid_search_NN.best_params_
multi_GradBoost_final, multi_NN_final = fine_tune_binary_models(x_train, y_train, x_val, y_val,
                                                                df_val_predictions_multi, pca,
                                                                params_gradBoost, params_NN)
# Add models into a voting ensemble
voting_clf_multi = voter(estimators=[('Logistic Regression', log_regress), ('gradBoost', multi_GradBoost_final),
                                     ('K_nearest', grid_search_kNearest),
                                     ('Neural network', multi_NN_final)], voting='hard')
voting_clf_multi.fit(x_train_processed, y_train)

# print ensemble results on training set
print("\n***** Multi, ensemble (training set)")
UtilityFunctions.model_results(voting_clf_multi, x_train_processed, y_train, df_val_predictions_multi,
                               model_name)

# Test ensemble on validation set
x_val_processed = process_data_existingPCA(x_val, pca)
print("\n***** Multi, ensemble (validation set)")
UtilityFunctions.model_results(voting_clf_multi, x_val_processed, y_val, df_val_predictions_multi, model_name)

print("\n\n***** Predictions from multi ensemble model on validation set")
print(df_val_predictions_multi)

# recombine training and validation and retrain
x_train_val_processed = process_data_existingPCA(x_train_val, pca)
voting_clf_multi.fit(x_train_val_processed, y_train_val)
print("\n***** Multi, ensemble training and val combined")
UtilityFunctions.model_results(voting_clf_multi, x_train_val_processed, y_train_val, df_val_predictions_multi,
                               model_name)
# Test ensemble on test set
print("\n***** Multi, ensemble Test set")
x_test_processed = process_data_existingPCA(x_test, pca)
UtilityFunctions.model_results(voting_clf_multi, x_test_processed, y_test, df_val_predictions_multi,
                               model_name)

# retrain on all data and make predictions on unseen data
x_all_processed = process_data_existingPCA(x_all, pca)
voting_clf_multi.fit(x_all_processed, y_all)
x_unlabelled = UtilityFunctions.read_unlabelled_CSV_data('XToClassifyMulti.csv')
x_unlabelled_processed = process_data_existingPCA(x_unlabelled, pca)
y_pred = voting_clf_multi.predict(x_unlabelled_processed)

print("\n\n----------------- Multi-class training finished ---------------------------------------------------------\n")

# write predictions to file
print("\nPredictions for unlabelled multi-class data\n")
file1 = open("PredictedClassesMulti.csv", "w")

print("Row,Prediction,Category")
file1.write("Row,Prediction,Category\n")
for i in range(0, len(y_pred)):
    string_line = str(i) + "," + str(y_pred[i]) + "," + str(key_multi[y_pred[i]])
    print(string_line)
    file1.write(string_line + "\n")
file1.close()


# measure time taken per observation
time_start = time.time()

# do PCA transform three times to ensure which represent real-world conidtions
x_single_example_processed = process_data_existingPCA(x_train_val.iloc[[0, 1, 2], :], pca)
x_single_example_processed = process_data_existingPCA(x_train_val.iloc[[0, 1, 2], :], pca)
x_single_example_processed = process_data_existingPCA(x_train_val.iloc[[0, 1, 2], :], pca)
voting_clf_multi.predict(x_single_example_processed)
time_end = time.time()

# then average over three obervations
print("\nAverage time taken for single observation (secs) ", (time_end - time_start) / 3)
