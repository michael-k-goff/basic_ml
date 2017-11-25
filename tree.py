# Decision tree code

from sklearn import tree

def DecisionTree(train_X, train_y, dev_X, dev_y, test_X, test_y, mss = 50):
    # Train the model on the training set
    clf = tree.DecisionTreeRegressor(min_samples_split=mss, random_state=99)
    clf = clf.fit(train_X,train_y)
    
    # The following generates a visual display of the decision tree.
    # tree.export_graphviz(clf,out_file='tree.dot') 
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('tree.png')
    
    return clf