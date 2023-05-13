from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

"""In this implementation, we define a sample dataset with text and labels. We separate the text and labels, 
and then create a bag-of-words representation of the text using the CountVectorizer() function from scikit-learn. We 
split the dataset into training and testing sets using the train_test_split() function, and train a Naive Bayes 
classifier using the MultinomialNB() class from scikit-learn. We test the classifier on the testing set and print the 
accuracy, and then predict the label of a new text using the trained classifier and the predict() method. """

# Training data
train_data = ["This book is good, very good.", "This novel is good.",
              "This book is some good and some bad.", "This novel is little good but little bad."]
train_labels = ['+', '+', "0", "0"]

# Test data
test_data = ["This book is bad, very bad.", 'This novel is bad', "This novel is little good but little bad."]
test_labels = ['+', '-', '0']

# Convert the text data into numerical data
vectorizer = CountVectorizer()
train_data_vectorized = vectorizer.fit_transform(train_data)
test_data_vectorized = vectorizer.transform(test_data)

# Train the Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(train_data_vectorized, train_labels)

# Predict on test data and evaluate accuracy
test_predictions = nb_classifier.predict(test_data_vectorized)
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy: ", accuracy * 100, "%")

# Convert the text data into numerical data using per-document binarization
vectorizer = CountVectorizer(binary=True)
train_data_vectorized = vectorizer.fit_transform(train_data)
test_data_vectorized = vectorizer.transform(test_data)

# Train the Naive Bayes model using BernoulliNB
nb_classifier = BernoulliNB()
nb_classifier.fit(train_data_vectorized, train_labels)

# Predict on test data and evaluate accuracy
test_predictions = nb_classifier.predict(test_data_vectorized)
print(test_predictions)
accuracy = accuracy_score(test_labels, test_predictions)
print("Accuracy: ", accuracy)
