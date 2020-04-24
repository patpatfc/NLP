#
import numpy as np
import pandas as pd

#
PATH = 'IMDB Dataset.csv'
df = pd.read_csv(PATH)
df.head()

#Önişlem
from bs4 import BeautifulSoup # To strip away the html without regex

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
remove_html(df.iloc[0][0])

#Kelime dağarcığı oluşturma
from tqdm import tqdm
max_obs = 20
observations = ''
for n in range(max_obs):
    observations += remove_html(df.iloc[n][0])

observations_clean = ','.join(e.lower() for e in observations.split() if e.isalnum())
set_of_words = set(observations_clean.split(','))

dict_list = []
for n in tqdm(range(len(df))):
    observation = remove_html(df.iloc[n][0]) # preprocessing step
    clean = ','.join(e.lower() for e in observation.split() if e.isalnum())
    dict_of_words = dict.fromkeys(set_of_words,0) # all words init at 0
    for word in clean.split(","):
        if word in dict_of_words:
            dict_of_words[word] += 1
    dict_list.append(dict_of_words)
    
pd.DataFrame(dict_list).sample(10)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

y = df.sentiment.replace({"positive":1,"negative":0})
x = df.review

bag = CountVectorizer()
X = bag.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X,y)

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train,y_train)

#Prediction
from sklearn.metrics import classification_report

preds = clf.predict(X_test)
print(classification_report(y_test,preds))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, preds)

#İnteraktif
test1 = "I really did not enjoy watching this. Very disappointed"
test2 = "What a wonderful movie. I enjoyed watching this with my kids"
clf.predict(bag.transform([test1]))
clf.predict(bag.transform([test2]))