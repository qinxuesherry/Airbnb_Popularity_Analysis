from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
import warnings
import matplotlib
warnings.filterwarnings("ignore")


# Set the backend
matplotlib.use('TkAgg')  # Set the backend
# Convert hex color to RGB
color_hex = "#fe595dff"
color_rgb = hex2color(color_hex)
# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("mycmap", [(1, 1, 1), color_rgb], N=256)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, hex2color
import seaborn as sns
import numpy as np
import pandas as pd
from itertools import cycle
import warnings
import matplotlib
warnings.filterwarnings("ignore")


## Create a class for the classifier methods
class Classifier:
    def __init__(self, params_lr=None, params_svm=None, params_lda=None, params_qda=None,params_rfc=None):
        self.model_lr = LogisticRegression(**(params_lr if params_lr is not None else {}), n_jobs=-1)
        self.model_svm = SVC(**(params_svm if params_svm is not None else {}))
        self.model_lda = LinearDiscriminantAnalysis(**(params_lda if params_lda is not None else {}))
        self.model_qda = QuadraticDiscriminantAnalysis(**(params_qda if params_qda is not None else {}))
        self.model_tree = tree.DecisionTreeClassifier()
        self.model_rfc = RandomForestClassifier(**(params_rfc if params_rfc is not None else {}))

    def fit(self, X_train, y_train, X_val,model_name):
        if model_name == 'lr':
            self.model_lr.fit(X_train, y_train)
            y_pred = self.model_lr.predict(X_val)
            y_prob = self.model_lr.predict_proba(X_val)
        elif model_name == 'svm':
            self.model_svm.fit(X_train, y_train)
            y_pred = self.model_svm.predict(X_val)
            y_prob = self.model_svm.predict_proba(X_val)
        elif model_name == 'lda':
            self.model_lda.fit(X_train, y_train)
            y_pred = self.model_lda.predict(X_val)
            y_prob = self.model_lda.predict_proba(X_val)
        elif model_name == 'qda':
            self.model_qda.fit(X_train, y_train)
            y_pred = self.model_qda.predict(X_val)
            y_prob = self.model_qda.predict_proba(X_val)
        elif model_name == 'tree':
            self.model_tree.fit(X_train, y_train)
            y_pred = self.model_tree.predict(X_val)
            y_prob = self.model_tree.predict_proba(X_val)
        elif model_name == 'rfc':
            self.model_rfc.fit(X_train, y_train)
            y_pred = self.model_rfc.predict(X_val)
            y_prob = self.model_rfc.predict_proba(X_val)
        else:
            raise ValueError("Invalid model name. Please choose from 'lr', 'svm', 'lda', or 'qda'.")

        return y_pred, y_prob

    def report(self, y_pred, y_val, y_prob):
        auc_score = roc_auc_score(y_val, y_prob, multi_class='ovr', average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        confusion = confusion_matrix(y_val, y_pred)
        percision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        support = np.unique(y_val, return_counts=True)[1]
        return auc_score, accuracy, report, confusion, percision, recall, f1, support


    def plot(self, confusion, y_prob, y_val,  save_fig=False, filename='model_plot.png'):
        # Initialize the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

        # Confusion Matrix
        sns.heatmap(confusion, annot=True, fmt='d', cmap=cmap, ax=ax1)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # ROC Curve
        n_classes = y_prob.shape[1]
        y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'black'])
        for i, color in zip(range(n_classes), colors):
            ax2.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ## After see all roc curve, set the ylim start from 0.6
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc='lower right')

        plt.tight_layout()
        if save_fig:
            plt.savefig(filename, format='png', dpi=300)  
        plt.show()

# Initialize the classifier parameters
params_lr={'max_iter': 1000, 'C': 0.5}
params_svm={'kernel': 'linear', 'C': 0.5, 'probability': True}
params_lda = {'solver': 'svd'}
params_qda = {'reg_param': 0.5}
params_rfc = {'max_depth': 5}


df = pd.read_csv('Data/cleaned.csv')
df.dropna(inplace=True)

# Use the first selected all 32 features
X = df.drop(columns=['popularity'])
y = df['popularity']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression Base Model 
model_lrg = Classifier(params_lr=params_lr)

## Fit the model
y_pred_lrg, y_prob = model_lrg.fit(X_train, y_train, X_test,'lr')
auc_score,lrg_accuracy, report, confusion, percision, recall, f1, support = model_lrg.report(y_pred_lrg, y_test, y_prob)

print("\nLogistic Regression")
print(f"AUC: {auc_score}")
print(f"Accuracy: {lrg_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

## Plot the confusion matrix and ROC curve
model_lrg.plot(confusion, y_prob, y_test,save_fig=True, filename='logistic_regression_analysis.png')



# Support Vector Machine Base Model
model_svm = Classifier(params_svm=params_svm)
y_pred_svm, y_prob = model_svm.fit(X_train, y_train, X_test,'svm')

auc_score,svm_accuracy, report, confusion, percision, recall, f1, support = model_svm.report(y_pred_svm, y_test, y_prob)

print("\nSVM")
print(f"AUC: {auc_score}")
print(f"Accuracy: {svm_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_svm.plot(confusion, y_prob, y_test,save_fig=True, filename='svm_analysis.png')



# Linear Discriminant Analysis Base Model
model_lda = Classifier(params_lda=params_lda)
y_pred_lda, y_prob = model_lda.fit(X_train, y_train, X_test,'lda')

auc_score,lda_accuracy, report, confusion, percision, recall, f1, support = model_lda.report(y_pred_lda, y_test, y_prob)

print("\nLDA")
print(f"AUC: {auc_score}")
print(f"Accuracy: {lda_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_lda.plot(confusion, y_prob, y_test,save_fig=True, filename='lda_analysis.png')



# Quadratic Discriminant Analysis Base Model
model_qda = Classifier(params_qda=params_qda)
y_pred_qda, y_prob = model_qda.fit(X_train, y_train, X_test,'qda')

auc_score,qda_accuracy, report, confusion, percision, recall, f1, support = model_qda.report(y_pred_qda, y_test, y_prob)

print("\nQDA")
print(f"AUC: {auc_score}")
print(f"Accuracy: {qda_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_qda.plot(confusion, y_prob, y_test,save_fig=True, filename='qda_analysis.png')





# Decision Tree Classifier Base Model
model_tree = Classifier()
y_pred_tree, y_prob = model_tree.fit(X_train, y_train, X_test,'tree')

auc_score,tree_accuracy, report, confusion, percision, recall, f1, support = model_tree.report(y_pred_tree, y_test, y_prob)

print("\nDecision Tree")
print(f"AUC: {auc_score}")
print(f"Accuracy: {tree_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_tree.plot(confusion, y_prob, y_test,save_fig=True, filename='tree_analysis.png')




# Random Forest Classifier Base Model
model_rfc = Classifier(params_rfc=params_rfc)
y_pred_rfc, y_prob = model_rfc.fit(X_train, y_train, X_test,'rfc')

auc_score,rfc_accuracy, report, confusion, percision, recall, f1, support = model_rfc.report(y_pred_rfc, y_test, y_prob)

print("\nRandom Forest")
print(f"AUC: {auc_score}")
print(f"Accuracy: {rfc_accuracy}")
print(f"Report: {report}")
print(f"Confusion Matrix: {confusion}")
print(f"Percision: {percision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Support: {support}")

model_rfc.plot(confusion, y_prob, y_test,save_fig=True, filename='rfc_analysis.png')