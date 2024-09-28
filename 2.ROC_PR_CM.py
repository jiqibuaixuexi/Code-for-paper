from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pandas as pd
import os
import argparse


def make_dir(dir):
    if not os.path.exist(dir):
        os.mkdir(dir)

        
def AUC_CI(auc, label, alpha = 0.05):
	label = np.array(label)
	n1, n2 = np.sum(label == 0), np.sum(label == 1) # n1:positive, n2:negative  
	q1 = auc / (2-auc)
	q2 = (2 * auc ** 2) / (1 + auc)
	se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 -1) * (q2 - auc ** 2)) / (n1 * n2))
	confidence_level = 1 - alpha
	z_lower, z_upper = norm.interval(confidence_level)
	lowerb, upperb = auc + z_lower * se, auc + z_upper * se
	return (lowerb, upperb)

def plot_AUC(ax, FPR, TPR, AUC, CI, label):
	label = '{}: {} ({}-{})'.format(str(label), round(AUC, 3), round(CI[0], 3), round(CI[1], 3))
	ax.plot(FPR, TPR, label = label)
	return ax

test_path = 'models_and_experts_output.csv'

# test data
test_result = pd.read_csv(test_path)

# gey true label and pred label
print(test_result)
true_label = test_result['true_label']
P_SRC_MT_pred_label = test_result['P_0.1_SRC_MT_pred_label']
P_upperbound_pred_label = test_result['P_upperbound_pred_label']
Expert_1_label = test_result['Expert_1']
Expert_2_label = test_result['Expert_2']
Expert_1_with_model_label = test_result['Expert_1_with_model']
Expert_2_with_model_label = test_result['Expert_2_with_model']

# print(true_label,P_SRC_MT_pred_label,P_upperbound_pred_label)

# P_upperbound
FPR_P_upperbound, TPR_P_upperbound, _ = roc_curve(true_label, P_upperbound_pred_label, pos_label = 1)
AUC_P_upperbound = roc_auc_score(true_label, P_upperbound_pred_label)
CI_P_upperbound = AUC_CI(AUC_P_upperbound, P_upperbound_pred_label, 0.05)
print('AUC_P_upperbound:',AUC_P_upperbound)

# P_SRC_MT
FPR_P_SRC_MT, TPR_P_SRC_MT, _ = roc_curve(true_label, P_SRC_MT_pred_label, pos_label = 1)
AUC_P_SRC_MT = roc_auc_score(true_label, P_SRC_MT_pred_label)
CI_P_SRC_MT = AUC_CI(AUC_P_SRC_MT, P_SRC_MT_pred_label, 0.05)
print('AUC_P_SRC_MT:',AUC_P_SRC_MT)

# choose style as ggplot
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax = plot_AUC(ax, FPR_P_SRC_MT, TPR_P_SRC_MT, AUC_P_SRC_MT, CI_P_SRC_MT, label = 'P_SRC_MT')
ax = plot_AUC(ax, FPR_P_upperbound, TPR_P_upperbound, AUC_P_upperbound, CI_P_upperbound, label = 'P_upperbound')


# Coordinate points for adding expert results 
# 1.Read the coordinates of the expert
expert_data = pd.read_csv('Experts_Metrics.csv',index_col=0)
print(expert_data)
# 2.Get the TPR and FPR of the expert after combining the model before and after combining the model as coordinates to mark on the ROC
FPR_1,FPR_2 = expert_data.loc['expert_1_label','FPR'],expert_data.loc['expert_2_label','FPR']
TPR_1,TPR_2 = expert_data.loc['expert_1_label','TPR'],expert_data.loc['expert_2_label','TPR']
FPR_3,FPR_4 = expert_data.loc['expert_1_with_model','FPR'],expert_data.loc['expert_2_with_model','FPR']
TPR_3,TPR_4 = expert_data.loc['expert_1_with_model','TPR'],expert_data.loc['expert_2_with_model','TPR']
# 3.Draw the coordinates of the expert on it
expert_1 = ax.scatter(FPR_1,TPR_1,marker='*',s=150,c='purple',label= 'expert 1')
expert_2 = ax.scatter(FPR_2,TPR_2,marker='p',s=150,c='orange',label= 'expert 2')
expert_3 = ax.scatter(FPR_3,TPR_3,marker='v',s=150,c='darkgreen',label= 'expert 1 + model')
expert_4 = ax.scatter(FPR_4,TPR_4,marker='^',s=150,c='brown',label= 'expert 2 + model')


# Plotting
# ax.plot((0, 1), (0, 1), ':', color = 'grey')
# ax.set_xlim(-0.01, 1.01)
# ax.set_ylim(-0.01, 1.01)
# ax.set_aspect('equal')
plt.title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

ax.legend()
plt.show()

# Plot the PR curve
from sklearn.metrics import precision_recall_curve,average_precision_score,plot_precision_recall_curve

# Calculate AP:average precision
# P_upperbound
precision_P_upperbound,recall_P_upperbound,thresholds_P_upperbound = precision_recall_curve(true_label,P_upperbound_pred_label)
AP_P_upperbound = average_precision_score(true_label,P_upperbound_pred_label)
# P_SRC_MT
precision_P_SRC_MT,recall_P_SRC_MT,thresholds_P_SRC_MT = precision_recall_curve(true_label,P_SRC_MT_pred_label)
AP_P_SRC_MT = average_precision_score(true_label,P_SRC_MT_pred_label)



print('AP_P_upperbound:',AP_P_upperbound,'\nAP_P_SRC_MT:',AP_P_SRC_MT)

# Plot
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax = plt.plot(recall_P_SRC_MT,precision_P_SRC_MT,label = 'Poolformer_SRC_MT')
ax = plt.plot(recall_P_upperbound,precision_P_upperbound,label = 'Poolformer_upperbound')



recall_1,recall_2 = expert_data.loc['expert_1_label','TPR'],expert_data.loc['expert_2_label','TPR']
recall_3,recall_4 = expert_data.loc['expert_1_with_model','TPR'],expert_data.loc['expert_2_with_model','TPR']
precision_1,precision_2 = expert_data.loc['expert_1_label','PPV'],expert_data.loc['expert_2_label','PPV']
precision_3,precision_4 = expert_data.loc['expert_1_with_model','PPV'],expert_data.loc['expert_2_with_model','PPV']

expert_1 = plt.scatter(recall_1,precision_1,marker='*',s=150,c='purple',label= 'expert 1')
expert_2 = plt.scatter(recall_2,precision_2,marker='p',s=150,c='orange',label= 'expert 2')
expert_3 = plt.scatter(recall_3,precision_3,marker='v',s=150,c='darkgreen',label= 'expert 1 + model')
expert_4 = plt.scatter(recall_4,precision_4,marker='^',s=150,c='brown',label= 'expert 2 + model')

plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Draw confusion matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)
	
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False) # 关闭网格线
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

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
    return ax

class_names = np.array(["AS","nonAS"]) 

# Poolformer_upperbound
plot_confusion_matrix(true_label, P_upperbound_pred_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Poolformer_upperbound') 
plt.show()

# SCDP
plot_confusion_matrix(true_label, P_SRC_MT_pred_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Poolformer_SRC_MT') 
plt.show()

# Expert_1
plot_confusion_matrix(true_label, Expert_1_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Expert 1') 
plt.show()

# Expert_2
plot_confusion_matrix(true_label, Expert_2_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Expert 2') 
plt.show()

# Expert_1_with_SCDP
plot_confusion_matrix(true_label, Expert_1_with_model_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Expert 1 with model')  # Poolformer_SRC_MT
plt.show()

# Expert_2_with_SCDP
plot_confusion_matrix(true_label, Expert_2_with_model_label, classes=class_names, normalize=True,title='Normalised Confusion Matrix of Expert 2 with model')  # Poolformer_SRC_MT
plt.show()
