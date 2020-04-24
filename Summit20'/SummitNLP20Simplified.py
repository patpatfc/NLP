import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#dataset yükle
df = pd.read_csv('IMDB Dataset.csv')

#Kelime dağarcığı
y = df.sentiment.replace({"positive":1,"negative":0})
x = df.review
bag = CountVectorizer()
X = bag.fit_transform(x)

#Train
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier(n_jobs = -1)
clf.fit(X_train, y_train)

#tahmin
preds = clf.predict(X_test)
cm = confusion_matrix(y_test, preds)

test1 = "I really did not enjoy watching this. Very disappointed"
test2 = "What a wonderful movie. I enjoyed watching this with my kids"
clf.predict(bag.transform([test1]))
clf.predict(bag.transform([test2]))