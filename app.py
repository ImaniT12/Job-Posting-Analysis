# Imports
import nltk
import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import plot_confusion_matrix
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def doc_preparer(doc, stop_words):
    '''
    
    :param doc: a document from the satire corpus 
    :return: a document string with words which have been 
            lemmatized, 
            parsed for stopwords, 
            made lowercase,
            and stripped of punctuation and numbers.
    '''
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in sw]
    # print(doc)
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)

# Title 
st.write("# Fraudulent Job Detection!")

# Opening intro text
st.write("# Is It Fake?")

description = st.text_area("Enter Job Description")
requirements = st.text_area("Enter Job Requirements")

words = description + ' ' + requirements

sw = stopwords.words('english')
caps_pattern = r"([A-Z]{3,})"


# Creating the dataframe to run predictions on
row = [description, requirements, words]
columns = ['description', 'requirements', 'words']

job = pd.DataFrame(dict(zip(columns, row)), index=[0])

job['words']

st.write(job)
#  Model Code - Change all things that say "data" to '"job"
job['words'] = job['words'].str.replace(caps_pattern, r" ").str.strip()

token_docsDR = [doc_preparer(doc, sw) for doc in words]

# cv = CountVectorizer()
# X_t_vecDR = cv.fit_transform(X_tDR)
# X_t_vecDR = pd.DataFrame.sparse.from_spmatrix(X_t_vecDR)
# X_t_vecDR.columns = sorted(cv.vocabulary_)
# X_t_vecDR.set_index(y_tDR.index, inplace=True)

# We then transform the validation set. (Do not refit the vectorizer!)
# X_val_vecDR = cv.transform(X_valDR)
# X_val_vecDR  = pd.DataFrame.sparse.from_spmatrix(X_val_vecDR)
# X_val_vecDR.columns = sorted(cv.vocabulary_)
# X_val_vecDR.set_index(y_valDR.index, inplace=True)

# mnb1 = MultinomialNB()

# mnb1.fit(X_t_vecDR, y_tDR)
# y_hatDR = mnb1.predict(X_val_vecDR)
# plot_confusion_matrix(mnb1, X_val_vecDR, y_valDR);

# st.write(f'accuracy score {accuracy_score(y_valDR, y_hatDR)}')
# st.write(f'precision score {precision_score(y_valDR, y_hatDR)}')

# Now predicting!
if st.button(label="Generate"):
    st.write(type(words))
    if type(words) != str:
        st.write("Invalid Input")
    else:
        
        load_vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
        # Load the model
        loaded_model = pickle.load(open("cv_model.sav", 'rb'))

        final= load_vectorizer.transform(token_docsDR)
        # Make predictions (and get out pred probabilities)
        pred = loaded_model.predict(final)[0]
        proba = loaded_model.predict_proba(final)[:,1][0]
        st.write(pred)
        
        # Sharing the predictions
        if pred == 0:
            st.write("### The job is not predicted to be fraudulent")
            st.write(f"Predicted probability to not be fraudulent: {proba*100:.2f}")

        elif pred == 1:
            st.write("### The job is predicted to be fraudulent!")
            st.write(f"Predicted probability to be fraudulent: {proba*100:.2f} %")
        pipe = make_pipeline(load_vectorizer, loaded_model)

        
        explainer = LimeTextExplainer(class_names=['Not Fraudulent','Fraudulent'])
        exp = explainer.explain_instance(words, 
                                        pipe.predict_proba, 
                                        num_features=20)
        st.write(exp.as_pyplot_figure())

