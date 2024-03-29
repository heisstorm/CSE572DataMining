# -* - coding: UTF-8 -* -
# ! /usr/bin/python
import matplotlib
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    combine = [train_df, test_df]
    train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)
    train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                          ascending=False)
    train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                          ascending=False)
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()



    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    pd.crosstab(train_df['Title'], train_df['Sex'])

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

    grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    guess_ages = np.zeros((2, 3))

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                    'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

        train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
        train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
                                                                                                  ascending=True)
        for dataset in combine:
            dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[dataset['Age'] > 64, 'Age']

        train_df = train_df.drop(['AgeBand'], axis=1)
        combine = [train_df, test_df]

        for dataset in combine:
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                        ascending=False)

        for dataset in combine:
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

        train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        combine = [train_df, test_df]

        for dataset in combine:
            dataset['Age*Class'] = dataset.Age * dataset.Pclass
        train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

        freq_port = train_df.Embarked.dropna().mode()[0]

        for dataset in combine:
            dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False)

        for dataset in combine:
            dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

        train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
        train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                                    ascending=True)
        for dataset in combine:
            dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
            dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
            dataset['Fare'] = dataset['Fare'].astype(int)

        train_df = train_df.drop(['FareBand'], axis=1)
        combine = [train_df, test_df]

        X_train = train_df.drop("Survived", axis=1)
        Y_train = train_df["Survived"]
        X_test = test_df.drop("PassengerId", axis=1).copy()

        # 1. logistic regression, still has Nan Value of Age, which will lead the Age*class to be 0, and falure of logistic regression, SVC

        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)
        Y_pred = logreg.predict(X_test.dropna(subset=['Age']))
        acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
        print(acc_log)

        # 2, SVC
        svc = SVC()
        svc.fit(X_train, Y_train)
        Y_pred = svc.predict(X_test.dropna(subset=['Age']))
        acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
        print(acc_svc)

        #3, KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test.dropna(subset=['Age']))
        acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
        print(acc_knn)

        # 4, Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian.fit(X_train, Y_train)
        Y_pred = gaussian.predict(X_test.dropna(subset=['Age']))
        acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
        print(acc_gaussian)

        # 5, Perceptron
        perceptron = Perceptron()
        perceptron.fit(X_train, Y_train)
        Y_pred = perceptron.predict(X_test.dropna(subset=['Age']))
        acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
        print(acc_perceptron)

        # 6, linear SVC
        linear_svc = LinearSVC()
        linear_svc.fit(X_train, Y_train)
        Y_pred = linear_svc.predict(X_test.dropna(subset=['Age']))
        acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
        print(acc_linear_svc)

        # 7, Stochastic Gradient Descent
        sgd = SGDClassifier()
        sgd.fit(X_train, Y_train)
        Y_pred = sgd.predict(X_test.dropna(subset=['Age']))
        acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
        print(acc_sgd)

        #8, Decision Tree
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        Y_pred = decision_tree.predict(X_test.dropna(subset=['Age']))
        acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
        print(acc_decision_tree)

        #9, Random Forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_test.dropna(subset=['Age']))
        random_forest.score(X_train, Y_train)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        print(acc_random_forest)

        models = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                      'Random Forest', 'Naive Bayes', 'Perceptron',
                      'Stochastic Gradient Decent', 'Linear SVC',
                      'Decision Tree'],
            'Score': [acc_svc, acc_knn, acc_log,
                      acc_random_forest, acc_gaussian, acc_perceptron,
                      acc_sgd, acc_linear_svc, acc_decision_tree]})
        models.sort_values(by='Score', ascending=False)
        print(models)