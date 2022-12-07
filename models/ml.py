#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
import numpy as np
import pandas as pd
import re
import string
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table("InsertTable",engine)


# In[4]:


df


# In[5]:


df.isnull().sum()


# In[6]:


X = df.message
Y = df.drop(['message','id','original','genre'],axis=1)
Y=Y.drop(['related','child_alone'],axis=1)


# In[7]:


Y.shape


# In[8]:


Y.mode()


# In[9]:


X=X.fillna(" ")


# In[10]:


Y=Y.fillna(Y.mode().iloc[0])


# In[11]:


X.isnull().sum()*100/len(df)


# 
# 
# 
# ### 2. Write a tokenization function to process your text data

# def tokenize(text):
#     """Tokenization function. Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
#     Returns tokenized text"""
#     
#     # Normalize text
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
#     
#     stop_words = stopwords.words("english")
#     
#     #tokenize
#     words = word_tokenize (text)
#     
#     #stemming
#     stemmed = [PorterStemmer().stem(w) for w in words]
#     
#     #lemmatizing
#     words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
#    
#     return words_lemmed

# In[12]:



def tokenize(text):
    text=re.sub(r"[^a-zA-Z]"," ",text.lower())    

    tokens = [word_tokenize(str(txt)) for txt in text]
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(str(tok)) for tok in tokens]
       

    return tokens
   


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[13]:


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
clf = MultiOutputClassifier(DecisionTreeClassifier())


# In[14]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(DecisionTreeClassifier()))
    ])


# 
# 
# 
# 
# 
# 
# 
# 
# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[15]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# In[16]:


pipeline.fit(X_train,Y_train)


# In[17]:


Y_pred=pipeline.predict(X_test)


# In[18]:


Y_pred


# In[19]:


col_name=Y.columns


# In[20]:


Y_predict=pd.DataFrame(Y_pred,columns=col_name)


# In[21]:


Y_predict


# In[ ]:





# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[22]:


print(classification_report(Y_test, Y_predict))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[23]:


clf.get_params()


# In[24]:



parameters=dict(vect__max_df=(0.75, 1.0)
                )
cv = GridSearchCV(pipeline,
                      param_grid=parameters,
                      scoring='accuracy',
                      n_jobs=-1,cv=5)
cv.fit(X_train, Y_train)
cv.best_params_


# In[25]:


Y_pred_cv=cv.predict(X_test)


# In[26]:


Y_predict_cv=pd.DataFrame(Y_pred,columns=col_name)


# In[27]:


Y_predict


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[28]:


print(classification_report(Y_test, Y_predict_cv))


# 

# In[ ]:





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[36]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# In[37]:


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2))
        }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)

    return cv


# In[38]:


def display_results(cv, Y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(Y_test, y_pred, labels=labels)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)


# In[ ]:


model = build_model()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
display_results(model, Y_test, y_pred)


# In[ ]:





# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




