import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("../dataset/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
X = df['message']
y = df['label']
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nEnter an SMS to classify as SPAM or HAM")
print("Type 'q' and press Enter to quit\n")
while True:
    user_message = input("Enter message: ")
    if user_message.lower() == 'q':
        print("Exiting... 👋")
        break
    user_message_vectorized = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vectorized)
    print("Prediction:", prediction[0])
    print("-" * 40)