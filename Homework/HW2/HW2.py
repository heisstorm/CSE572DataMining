import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    # Load the training data
    train_data = pd.read_csv('train.csv')

    # Display the first few rows of the training data to understand its structure
    print(train_data.head())

    # Fill missing values
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

    # Select features and target variable
    X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = train_data['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the decision tree model
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Predict on the test set
    y_pred = decision_tree.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # print(accuracy)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(decision_tree,
              filled=True,
              rounded=True,
              class_names=['Not Survived', 'Survived'],
              feature_names=X.columns.tolist(),
              max_depth=5,  # Limit depth for visualization purposes
              fontsize=10)
    plt.show()



    # Apply five-fold cross-validation to the decision tree model
    cv_scores = cross_val_score(decision_tree, X, y, cv=5)

    # Calculate the average classification accuracy
    average_accuracy = cv_scores.mean()
    # print(average_accuracy)


    # Initialize and fine-tune the random forest model
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Apply five-fold cross-validation to the random forest model
    cv_scores_rf = cross_val_score(random_forest, X, y, cv=5)

    # Calculate the average classification accuracy for the random forest model
    average_accuracy_rf = cv_scores_rf.mean()
    print(average_accuracy_rf)
