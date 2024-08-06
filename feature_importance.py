import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,classification_report
import seaborn as sns 
from matplotlib.colors import LinearSegmentedColormap, hex2color
from matplotlib.colors import to_rgba



# Convert hex color to RGB
color_hex = "#fe595dff"
color_rgb = hex2color(color_hex)

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("mycmap", [(1, 1, 1), color_rgb], N=256)



data = pd.read_csv('Data/cleaned.csv')

X = data.drop(columns=['popularity'])
Y = data['popularity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


## Decision Tree Model
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)

yp_train = model.predict(X_train)
yp_test = model.predict(X_test)

# Define function to calculate the accuracy, precision and recall
def confusion_plot(Y_test, Y_pred):
    matrix = confusion_matrix(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='weighted', zero_division=0)
    recall = recall_score(Y_test, Y_pred, average='weighted', zero_division=0)
    print("Accuracy: ", accuracy)
    print("Weighted Precision:", precision) 
    print("Weighted Recall:", recall)
    print(matrix)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, cbar=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Accurancy, precision and recall of plain decision tree
print("------TRAINING------")
confusion_plot(y_train,yp_train)
print("------TEST------")
confusion_plot(y_test,yp_test)

# Visualize plain tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True)
plt.title("Decision Tree visualized")
plt.show()

# Feature importance Plot Decision Tree
importances_tree = model.feature_importances_

features = ['host_response_time', 'host_response_rate', 'host_is_superhost',
       'room_type', 'accommodates', 'minimum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_value', 'instant_bookable', 'reviews_per_month',
       'host_years', 'review_years_range']

# Sorting the features based on their importance
indices = np.argsort(importances_tree)[::-1]

# Convert hex color to RGBA
base_color = to_rgba("#fe595dff")

# Generate shades of the base color
colors = [tuple(np.array(base_color) * (1 - (0.5 * i / len(features)))) for i in range(len(features))]

# Create the plot
plt.figure(figsize=(8, 6),facecolor='none')
plt.title('Decision Tree Feature Importances')
plt.bar(range(X.shape[1]), importances_tree[indices], color=colors, align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


# Random Forest Model
model_rf = RandomForestClassifier()
model_rf = model_rf.fit(X_train, y_train)

rf_yp_test = model_rf.predict(X_test)
rf_yp_train = model_rf.predict(X_train)

# Calculate the class probabilities based on their distribution in the training set
class_probabilities = y_train.value_counts(normalize=True)

# Generate weighted random predictions based on class distribution
weighted_random_train_prediction = np.random.choice(class_probabilities.index, size=len(y_train), p=class_probabilities.values)
weighted_random_test_predictions = np.random.choice(class_probabilities.index, size=len(y_test), p=class_probabilities.values)

# Accurancy, precision and recall of random forest
print("------TRAINING------")
confusion_plot(y_train,rf_yp_train)
print("------TEST------")
confusion_plot(y_test,rf_yp_test)


# Feature importance Plot RFC
importances = model_rf.feature_importances_

# Feature names for x-axis labels
features = ['host_response_time', 'host_response_rate', 'host_is_superhost',
       'room_type', 'accommodates', 'minimum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_value', 'instant_bookable', 'reviews_per_month',
       'host_years', 'review_years_range']

# Sorting the features based on their importance
indices = np.argsort(importances)[::-1]

# Convert hex color to RGBA
base_color = to_rgba("#fe595dff")

# Generate shades of the base color
colors = [tuple(np.array(base_color) * (1 - (0.5 * i / len(features)))) for i in range(len(features))]

# Create the plot
plt.figure(figsize=(8, 6),facecolor='none')
plt.title('Random Forest Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color=colors, align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()