import re, collections, warnings, enchant
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# Load the dataset
data = pd.read_excel('/home/abhinavgorantla/hdd/ai_flask/app/training.xlsx')
essay_set_num = 7
data = data.loc[data['essay_set'] == essay_set_num]
print(data)

#Filtering the dataset
data.drop(data.iloc[:, 1:2], inplace=True, axis=1)
data.drop(data.iloc[:, 2:5], inplace=True, axis=1)
data.drop(data.iloc[:, 3:], inplace=True, axis=1)
data.reset_index(drop=True, inplace=True)
num_essays = data.shape[0]




def get_wordlist(sentence):
    # Remove non-alphanumeric characters
    sentence = re.sub("[^a-zA-Z0-9]"," ", sentence)
    words = nltk.word_tokenize(sentence)

    return words

def get_tokenized_sentences(essay):
    sentences = nltk.sent_tokenize(essay.strip())
    
    tokenized_sentences = []
    for sentence in sentences:
        if len(sentence) > 0:
            tokenized_sentences.append(get_wordlist(sentence))
    
    return tokenized_sentences

def get_word_length_average(essay):
    # Sanitize essay
    essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(essay)
    avg = sum(len(word) for word in words) / len(words)
    
    return avg


def get_word_count(essay):
    essay = re.sub(r'\W', ' ', essay)
    count = len(nltk.word_tokenize(essay))
    
    return count

def get_sentence_count(essay):
    sentences = nltk.sent_tokenize(essay)
    count = len(sentences)

    return count

def get_lemma_count(essay):
    tokenized_sentences = get_tokenized_sentences(essay)      
    
    lemmas = []
    for sentence in tokenized_sentences:
        pos_tagged_tokens = nltk.pos_tag(sentence) 
        for token_tuple in pos_tagged_tokens:
            word = token_tuple[0]
            pos_tag = token_tuple[1]
            # assume default part of speech to be noun
            pos = wordnet.NOUN
            if pos_tag.startswith('J'):
                pos = wordnet.ADJ
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                
            lemmas.append(WordNetLemmatizer().lemmatize(word, pos))
    
    lemma_count = len(set(lemmas))
    
    return lemma_count

def get_pos_counts(essay):
    tokenized_sentences = get_tokenized_sentences(essay)
    
    nouns, adjectives, verbs, adverbs = 0, 0, 0, 0
    
    essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(essay)
    
    for sentence in tokenized_sentences:
        pos_tagged_tokens = nltk.pos_tag(sentence)
        for token_tuple in pos_tagged_tokens:
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'): 
                nouns += 1
            elif pos_tag.startswith('J'):
                adjectives += 1
            elif pos_tag.startswith('V'):
                verbs += 1
            elif pos_tag.startswith('R'):
                adverbs += 1
    
    return nouns/len(words), adjectives/len(words), verbs/len(words), adverbs/len(words)



def get_spell_error_count(essay):   
    essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(essay)
    
    d = enchant.Dict("en_US")
    misspelt_count = 0
    for word in words:
        if(d.check(word) == False):
            misspelt_count += 1
    
    total_words = get_word_count(essay)
    error_prob = misspelt_count/total_words
    
    return error_prob

def get_sentiment_tags(essay):
    negative, positive, neutral = 0, 0, 0
    
    ss = SentimentIntensityAnalyzer().polarity_scores(essay)
    for k in sorted(ss):
        if k == 'compound':
            pass
        elif k == 'neg':
            negative += ss[k]
        elif k == 'pos':
            positive += ss[k]
        elif k == 'neu':
            neutral += ss[k]
            
    return negative, positive, neutral

def get_tfidf_vectors(essays):
    vectorizer = TfidfVectorizer(stop_words='english')
    
    words = []
    for essay in essays:
        essay = re.sub(r'\W', ' ', essay)
        words.append(nltk.word_tokenize(essay))
        
    docs_lemmatized = [[WordNetLemmatizer().lemmatize(j) for j in i]for i in words]
    
    corpus = [' '.join(i) for i in docs_lemmatized]
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, vectors

feature_names,vectors_all = get_tfidf_vectors(data['essay'])
print("num of essays X number of features",vectors_all.shape)

# SVD represent documents and terms in vectors 
reduced_dim = vectors_all.shape[0]
svd_model = TruncatedSVD(n_components=reduced_dim, algorithm='randomized', random_state=122)
lsa = svd_model.fit_transform(vectors_all)

print(pd.DataFrame(svd_model.components_,index=range(reduced_dim), columns=feature_names))

# Compute document similarity using LSA components
lsa_similarity = np.asarray(np.asmatrix(lsa) * np.asmatrix(lsa).T)
pd.DataFrame(lsa_similarity,index=range(num_essays), columns=range(num_essays))



highest = max(data['domain1_score'].tolist())

def get_cosine_similarity(essay_id):
    index_high = data.index[data['domain1_score'] == highest].tolist()
    n = len(index_high)

    j = data.index[data['essay_id'] == essay_id]
    similarity = 0
    for i in index_high:
        similarity += cosine_similarity(vectors_all[i,:],vectors_all[j,:])
    similarity /= n
    
    return np.asscalar(similarity)




def extract_features(data):
    
    features = data.copy()
    print(feature_names)
    
    features['word_count'] = features['essay'].apply(get_word_count)
    print("Added 'word_count' feature successfully.")
    print(type(features['essay']), features['essay'])
    
    features['sent_count'] = features['essay'].apply(get_sentence_count)
    print("Added 'sent_count' feature successfully.")
    
    features['avg_word_len'] = features['essay'].apply(get_word_length_average)
    print("Added 'avg_word_len' feature successfully.")
    
    features['lemma_count'] = features['essay'].apply(get_lemma_count)
    print("Added 'lemma_count' feature successfully.")
    
    features['spell_err_count'] = features['essay'].apply(get_spell_error_count)
    print("Added 'spell_err_count' feature successfully.")
    
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(get_pos_counts))
    print("Added 'noun_count', 'adj_count', 'verb_count' and 'adv_count' features successfully.")
    
    features['neg_score'], features['pos_score'], features['neu_score'] = zip(*features['essay'].map(get_sentiment_tags))
    print("Added 'neg_score', 'pos_score' and 'neu_score' features successfully.")
    
    features['cosine_similarity'] = features['essay_id'].apply(get_cosine_similarity)
    print("Added 'similarity' feature successfully.")
        
    # TODO: LSA 
    
    return features

features_set1 = extract_features(data)

filename = 'features_set_' + str(essay_set_num) + '.csv'
features_set1.to_csv(filename, index=False)