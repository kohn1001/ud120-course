from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):   
    '''Create and fit a Gaussian classifier of features and labels
    Args:
        features_train: 
            numpy array of features
        labels_train:
            numpy array of labels for the features

    Returns:
        clf:
            a Gaussian Naive Bayes classifier, trained with the training data
    '''
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf