from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from vectorization.data_splitting import train_X, train_y, test_X, test_y

vectorizer = CountVectorizer(ngram_range=(1, 4)).fit(train_X)
transformed_train_X = vectorizer.transform(train_X)
transformed_test_X = vectorizer.transform(test_X)


def get_LR_score():
    """Logistic Regression algorithm score."""
    modelLR = LogisticRegression(C=100, max_iter=512).fit(transformed_train_X, train_y)
    predictionsLR = modelLR.predict(transformed_test_X)
    return f1_score(test_y, predictionsLR)


def get_NB_score():
    """Multinomial Naive Bayes algorithm score."""
    modelNB = MultinomialNB(alpha=1.7).fit(transformed_train_X, train_y)
    predictionsNB = modelNB.predict(transformed_test_X)
    return f1_score(test_y, predictionsNB)


def get_RFC_score():
    """Random Forest Classifier algorithm score."""
    modelRFC = RandomForestClassifier(n_estimators=20).fit(transformed_train_X, train_y)
    predictionsRFC = modelRFC.predict(transformed_test_X)
    return f1_score(test_y, predictionsRFC)


def get_SGDC_score():
    """Stochastic Gradient Descent Classifier algorithm score."""
    modelSGDC = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3).fit(
        transformed_train_X, train_y
    )
    predictionsSGDC = modelSGDC.predict(transformed_test_X)
    return f1_score(test_y, predictionsSGDC)


def get_SVC_score():
    """Support Vector Classifieralgorithm score."""
    modelSVC = SVC(C=100).fit(transformed_train_X, train_y)
    predictionsSVC = modelSVC.predict(transformed_test_X)
    return f1_score(test_y, predictionsSVC)


def run():
    print()
    print("Fetching F1 Scores for different Algorithms...")

    print()
    print("Logistic Regression...")
    print(get_LR_score())

    print()
    print("Multinomial Naive Bayes...")
    print(get_NB_score())

    print()
    print("Random Forest Classifier...")
    print(get_RFC_score())

    print()
    print("Stochastic Gradiant Descent Classifier")
    print(get_SGDC_score())

    print()
    print("Support Vector Classification...")
    print(get_SVC_score())

    print()
    print("END!")
