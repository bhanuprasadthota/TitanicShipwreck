# Assignment 02 - CIS 530  Titanic shipwreck


# Importing necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import GridSearchCV

# Loading the training and testing data
train_data = pd.read_csv('/Users/bhanuprasadthota/Downloads/train.csv')
test_data = pd.read_csv('/Users/bhanuprasadthota/Downloads/test.csv')

# Display the first few rows of the training data
#print(train_data.head())

# Data Preprocessing and Handle missing values
train_data.fillna(method='ffill', inplace=True)
test_data.fillna(method='ffill', inplace=True)

# Encoding categorical variables (Sex and Embarked)
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

# Selecting features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_data['Survived']

# Creating a decision tree model
model = DecisionTreeClassifier()

# Fit the model to the entire training data
model.fit(X, y)

# Finding the maximum depth and minimum leafs in the decision tree
param_grid = {
    'max_depth': list(range(1, 31)),
    'min_samples_leaf': list(range(1, 31))
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
max_depth = best_params['max_depth']
min_samples_leaf = best_params['min_samples_leaf']

#print(max_depth)
#print(min_samples_leaf)

# Strategies to Prevent Overfitting:
# 1. Limiting Tree Depth
model_depth = DecisionTreeClassifier(max_depth=6)
model_depth.fit(X, y)

# 2. Minimum Samples per Leaf Node
model_min_samples = DecisionTreeClassifier(min_samples_leaf=10)
model_min_samples.fit(X, y)

# 3. Feature Selection
X_selected_features = train_data[['Pclass', 'Sex', 'Age', 'Fare']]
model_selected_features = DecisionTreeClassifier()
model_selected_features.fit(X_selected_features, y)

# Training the final model with the best strategy
final_model = DecisionTreeClassifier(max_depth=6)
final_model.fit(X, y)

# Calculating and printing the accuracy on the training data
y_pred = final_model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy on the training data: {accuracy}")

# Visualizing the decision tree
dot_data = export_graphviz(final_model, out_file=None,
                           feature_names=X.columns.tolist(),
                           class_names=["Did not Survive", "Survived"],
                           filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data, format="png")
graph.render("titanic_decision_tree")

# Preparing test data
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Making predictions on the test data
test_predictions = final_model.predict(X_test)

# Creating a DataFrame with PassengerId and Survived columns
prediction = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
#print(submission)

# Saving the submission to a CSV file
prediction.to_csv('titanic_survival_prediction.csv', index=False)
