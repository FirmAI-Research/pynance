# run multiple models and return the most accurate model
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pydotplus
from sklearn import tree


def learn(df):
    """[Summary]
    Iterate through ml models to find the best one for a given dataset
        Assumes non numeric data is already encodeds
    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    df = df.dropna(how='any', axis=0)

    dep_vars = ['p1_result','p2_result', 'score']
    features = [c for c in df.columns if c not in dep_vars]

    X = df[features]        # independent variables  
    y = df['p1_result']     # dependant variable

    X_train, X_test, y_train, y_test = train_test_split(X, df.p1_result, test_size=0.3,random_state=11) # 70% training and 30% test

    models = [LogisticRegression(random_state=100),  svm.SVC(kernel='linear'), KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier() ]

    for model in models:

        print(f'*** {model} ***')

        clf = model.fit(X_train, y_train)

        yhat = clf.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, yhat)) #  Model Accuracy: how often is the classifier correct?
        print("Precision:",metrics.precision_score(y_test, yhat))  # Model Precision: what percentage of positive tuples are labeled as such?
        print("Recall:",metrics.recall_score(y_test, yhat)) # Model Recall: what percentage of positive tuples are labelled as such?

        if isinstance(model, DecisionTreeClassifier):
            scores = cross_val_score(estimator=clf, X=X_test, y=y_test, cv=10)
            print(scores)

            data = tree.export_graphviz(clf, out_file=None, feature_names=X_test.columns)
            graph = pydotplus.graph_from_dot_data(data)
            print(graph)
            graph.write_png('models/mydecisiontree.png')