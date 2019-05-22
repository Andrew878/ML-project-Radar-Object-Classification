import numpy as np
import sklearn as sk
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# global variable to print graph
global PLOT_GRAPHS
PLOT_GRAPHS = True

# dictionaries for classification
global key_binary
key_binary = {0: 'book', 1: 'plastic case'}
global key_multi
key_multi = {0: 'air', 1: 'book', 2: 'hand', 3: 'knife', 4: 'plastic case'}


def get_column_names():
    """Computes the column name of each feature"""

    global column_names, column_name
    CHANNEL_SIZE = 64
    NUM_MEASURE = 3
    column_names = []
    for i in range(1, 769):
        feature = i
        component = (i - 1) // NUM_MEASURE + 1
        channel = (i - 1) // (CHANNEL_SIZE * NUM_MEASURE) + 1

        if (i % 3 == 0):
            description = 'max'

        elif (i % 2 == 0):
            description = 'min'
        else:
            description = 'mean'

        column_name = str(feature) + "," + str(description) + "," + str(component) + "," + str(channel)
        column_names.append(column_name)

    return column_names


def read_CSV_data(x_filename, y_filename):
    """Reads the data from the input filenames, stores in a panda, and then performs test/train/validation split"""

    # get column names
    column_names = get_column_names()

    # read files
    x_all = pd.read_csv(x_filename, header=None, names=column_names)
    y_all = pd.read_csv(y_filename, header=None, names=['Y'])

    # check for null values
    is_null_X = x_all.isna()
    is_null_Y = y_all.isna()
    for column_name in column_names:
        if (True in is_null_X[column_name].values):
            print("Null value in ", column_name)
    if (True in is_null_Y.values):
        print("Null value in Y")

    # split into test, val and training sets. Constant random seed to ensure reproducibility
    x_train_val, x_test, y_train_val, y_test = sk.model_selection.train_test_split(x_all, y_all, test_size=0.10,
                                                                                   random_state=20, stratify=y_all)
    x_train, x_val, y_train, y_val = sk.model_selection.train_test_split(x_train_val, y_train_val, test_size=0.10,
                                                                         random_state=20)

    # plot correlation heat-map of features
    plot_correlation_heatmap(x_train, 'Correlation heat-map of 768 original features \n(individual labels not shown)')

    return x_train_val, x_test, y_train_val, y_test, x_train, x_val, y_train, y_val, x_all, y_all;


def read_unlabelled_CSV_data(x_filename):
    """Read the unlabelled CSV files. Converts into pandas """

    x_all = pd.read_csv(x_filename, header=None, names=column_names)
    is_null_X = x_all.isna()

    # check no missing values
    for column_name in column_names:
        if (True in is_null_X[column_name].values):
            print("Null value in ", column_name)

    return x_all;


def plot_correlation_heatmap(d, name, linewidths=0.0001):
    """Plots a correlation heat-map graph. Code adapted from Seaborn website. Accessed 4/04/2019.
    https://seaborn.pydata.org/examples/many_pairwise_correlations.html"""

    sns.set(style="white")

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=linewidths, cbar_kws={"shrink": .5}, xticklabels=False, yticklabels=False)
    f.suptitle(name, fontsize=16)
    if PLOT_GRAPHS:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`

    Code taken from sklearn website. Accessed 4/04/2019.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"""

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes =classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if PLOT_GRAPHS:
        plt.show()

    return ax





def model_results(model, x_val_set, y_val_set, val_predictions, name_of_model):
    """This function presents the results of a model and calculates some performance measures."""

    # # Make predictions using the testing set
    y_pred_val = model.predict(x_val_set)

    # print key model data
    print("Model: " + name_of_model)

    # print performance data
    print('Accuracy: %.3f'
          % accuracy_score(y_val_set, y_pred_val))

    print('Precision (macro): %.3f'
          % precision_score(y_val_set, y_pred_val, average='macro'))

    print('Recall (macro): %.3f' % recall_score(y_val_set, y_pred_val, average='macro'))

    print('F1 score (macro): %.3f' % f1_score(y_val_set, y_pred_val, average='macro'))

    val_predictions[name_of_model] = pd.DataFrame(y_pred_val)


    list_class = [int(i) for i in y_val_set.values]
    set_class = set(list_class)
    if(len(set_class)==2):
        list_class = [int(i) for i in range(0,2)]
        list_class = [key_binary[int(i)] for i in range(0,2)]
    elif (len(set_class)==5):
        list_class = [int(i) for i in range(0,5)]
        list_class = [key_multi[int(i)] for i in range(0,5)]



    np.set_printoptions(precision=4)
    plot_confusion_matrix(y_val_set, y_pred_val, classes=list_class, normalize=True,
                          title="Normalized confusion matrix for " + name_of_model)
