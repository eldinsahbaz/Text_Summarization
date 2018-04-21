import pandas as pd, numpy as np, tensorflow as tf, re, time, sys, contractions, _pickle as pickle, os, nltk, random
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.python.layers.core import Dense
from nltk.corpus import stopwords
from collections import Counter
from pprint import pprint

def filter_symbols(input_summary, input_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub(r'https?:\/\/.*[\r\n]*', '', zero(text), flags=re.MULTILINE)
        two = lambda text: re.compile(r'(<!--.*?-->|<[^>]*>)').sub('', one(text))
        three = lambda text: re.sub(r'&amp;', '', two(text))
        four = lambda text: re.sub('\.\.\.', '.', three(text))
        five = lambda text: [nltk.word_tokenize(re.sub(r'[^a-zA-Z ]+', '', tokens)) for tokens in four(text).split('.')]

        return (five(input_summary), [[lemmatizer.lemmatize(word2) for word2 in word1  if (word2 not in stop_words)] for word1 in five(input_text)])
    except:
        return None

def filter_symbols_test(input_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    try:
        zero = lambda text: contractions.fix(text.lower())
        one = lambda text: re.sub(r'https?:\/\/.*[\r\n]*', '', zero(text), flags=re.MULTILINE)
        two = lambda text: re.compile(r'(<!--.*?-->|<[^>]*>)').sub('', one(text))
        three = lambda text: re.sub(r'&amp;', '', two(text))
        four = lambda text: re.sub('\.\.\.', '.', three(text))
        five = lambda text: [nltk.word_tokenize(re.sub(r'[^a-zA-Z ]+', '', tokens)) for tokens in four(text).split('.')]

        return [[lemmatizer.lemmatize(word2) for word2 in word1  if (word2 not in stop_words)] for word1 in five(input_text)]
    except:
        return None

def clean_data(path, target):
    cleaned = None
    try:
        with open(target, 'rb') as file:
            cleaned = pickle.loads(file.read())
    except:
        data = [tuple(x) for x in pd.read_csv('Reviews.csv')[['Summary', 'Text']].values.tolist()]
        pool = Pool()

        cleaned = pool.starmap(filter_symbols, data)

        pool.close()
        pool.join()

        cleaned = list(filter(lambda y: y, cleaned))

        with open(target, 'wb') as file: pickle.dump(cleaned, file)

    return cleaned

def create_embeddings(data, cutoff, embedding_map_target, embedding_summary_target, embedding_review_target):
    DNS = {'forward':{'<UNK>':0, '<PAD>':1, '<EOS>':2, '<GO>':3}, 'backward':{0:'<UNK>', 1:'<PAD>', 2:'<EOS>', 3:'<GO>'}}
    words = list()
    embedded_summaries, embedded_reviews = list(), list()
    plaintext_summaries, plaintext_reviews = list(), list()

    #create mapping for word -> int and for int -> word
    for summary, review in data:
        plaintext_summaries.append(sum(summary, []))
        plaintext_reviews.append(sum(review, []))
        words.extend(sum(summary, []))
        words.extend(sum(review, []))

    word_frequencies = [x for x in sorted(Counter(words).items(), key=lambda x: x[1], reverse=False) if (x[1] >= cutoff)]

    if word_frequencies:
        words, freqs = list(zip(*word_frequencies))
        DNS['forward'].update(dict(zip(words, list(range(len(DNS['forward']), len(words)+len(DNS['forward']))))))
        DNS['backward'].update({v: k for k, v in DNS['forward'].items()})

    #Compute the translation to int for the full text
    for summary in plaintext_summaries:
        temp_summary = list()
        temp_summary.append(DNS['forward']['<GO>'])
        for word in summary:
            try: temp_summary.append(DNS['forward'][word])
            except : temp_summary.append(DNS['forward']['<UNK>'])
        temp_summary.append(DNS['forward']['<EOS>'])
        embedded_summaries.append(temp_summary)

    for review in plaintext_reviews:
        temp_summary = list()
        temp_summary.append(DNS['forward']['<GO>'])
        for word in review:
            try: temp_summary.append(DNS['forward'][word])
            except : temp_summary.append(DNS['forward']['<UNK>'])
        temp_summary.append(DNS['forward']['<EOS>'])
        embedded_reviews.append(temp_summary)

    #Compute the truncated version of the texts above
    summary_lengths, review_lengths, review_unk_counts, summary_unk_counts = list(), list(), list(), list()

    for sentence in embedded_summaries: summary_lengths.append(len(sentence))
    summary_pd = pd.DataFrame(summary_lengths, columns=['counts'])
    max_summary_length = int(np.percentile(summary_pd.counts, 90))

    for sentence in embedded_reviews: review_lengths.append(len(sentence))
    review_pd = pd.DataFrame(review_lengths, columns=['counts'])
    max_review_length = int(np.percentile(review_pd.counts, 90))

    data_pd = pd.DataFrame(summary_lengths+review_lengths, columns=['counts'])
    min_length = int(np.percentile(data_pd.counts, 5))

    for sentence in embedded_reviews: review_unk_counts.append(Counter(sentence)[DNS['forward']['<UNK>']])
    review_pd = pd.DataFrame(review_unk_counts, columns=['counts'])
    unk_review_limit = int(np.percentile(review_pd.counts, 5))

    for sentence in embedded_summaries: summary_unk_counts.append(Counter(sentence)[DNS['forward']['<UNK>']])
    review_pd = pd.DataFrame(summary_unk_counts, columns=['counts'])
    unk_summary_limit = int(np.percentile(review_pd.counts, 5))

    truncated_summaries, truncated_reviews = list(), list()
    for summary in embedded_summaries:
        temp = summary[:max_summary_length]
        temp[-1] = DNS['forward']['<EOS>']
        if len(temp) < max_summary_length: temp[0:0] = [DNS['forward']['<PAD>']]*(max_summary_length-len(temp))
        truncated_summaries.append(temp)

    for review in embedded_reviews:
        temp = review[:max_review_length]
        temp[-1] = DNS['forward']['<EOS>']
        if len(temp) < max_review_length: temp[0:0] = [DNS['forward']['<PAD>']]*(max_review_length-len(temp))
        truncated_reviews.append(temp)

    cleaned_truncated_summaries, cleaned_truncated_reviews = list(), list()
    for summary, review in list(zip(embedded_summaries, embedded_reviews)):
        summary_count, review_count = Counter(summary), Counter(review)

        if ((summary_count[DNS['forward']['<UNK>']] <= unk_summary_limit) and (review_count[DNS['forward']['<UNK>']] <= unk_review_limit) and (len(summary) >= min_length) and (len(review) >= min_length)):
            cleaned_truncated_summaries.append(summary)
            cleaned_truncated_reviews.append(review)

    #print(DNS)
    #print(list(zip(*data)))
    #print(embedded_summaries)
    #print(embedded_reviews)
    #print(cleaned_truncated_summaries)
    #print(cleaned_truncated_reviews)
    #input()
    
    #Save files
    with open(embedding_map_target, 'wb') as file:
        pickle.dump(DNS, file)

    with open(embedding_summary_target, 'wb') as file:
        pickle.dump(embedded_summaries, file)

    with open(embedding_review_target, 'wb') as file:
        pickle.dump(embedded_reviews, file)

    with open("TRUNCATED_" + embedding_summary_target, 'wb') as file:
        pickle.dump(cleaned_truncated_summaries, file)

    with open("TRUNCATED_" + embedding_review_target, 'wb') as file:
        pickle.dump(cleaned_truncated_reviews, file)

    return (DNS, embedded_summaries, embedded_reviews, max_summary_length, max_review_length, min_length, unk_review_limit, unk_summary_limit, unk_summary_limit)

#https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
#https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/
def build_model(review_length, summary_length, vocab_size):
    return None

def train(xTrain, xTest, yTrain, yTest, modelDir, modelFileName):
    return None

def prep_test_data(text, DNS, max_length):
    temp_text = list()
    cleaned_test = filter_symbols_test(text)
    temp_text.append(DNS['forward']['<GO>'])

    for word in text:
        try: temp_text.append(DNS['forward'][word])
        except: temp_text.append(DNS['forward']['<UNK>'])

    temp_text.append(DNS['forward']['<EOS>'])

    temp_text = temp_text[:max_length]
    temp_text[-1] = DNS['forward']['<EOS>']
    if len(temp_text) < max_length: temp_text[0:0] = [DNS['forward']['<PAD>']]*(max_length-len(temp_text))
    
    return temp_text

def test(text, DNS, modelDir):
    return None

output_params = None
with tf.device('/device:GPU:0'):
    if (__name__ == '__main__') and (len(sys.argv) > 1):
        if 'train' == sys.argv[1]:
            cutoff = 15
            (DNS, embedded_summaries, embedded_reviews, max_summary_length, max_review_length, min_length, unk_review_limit, unk_summary_limit, unk_summary_limit) = create_embeddings(clean_data('Reviews.csv', 'cleaned_2.txt'), cutoff, "Embedding_Map.txt", "Embedded_Summary.txt", "Embedded_Review.txt")
            output_params = (DNS, embedded_summaries, embedded_reviews, max_summary_length, max_review_length, min_length, unk_review_limit, unk_summary_limit, unk_summary_limit)
            split_position = int(len(embedded_summaries)*0.8)
            xTrain, xTest = embedded_reviews[:split_position], embedded_reviews[split_position:]
            yTrain, yTest = embedded_summaries[:split_position], embedded_summaries[split_position:]
            
            model = train(xTrain, xTest, yTrain, yTest, "/home/esahbaz/Computer/Users/sahba/PycharmProjects/NLP_Project/model", "Seq2Seq.model")
        
        elif 'test' == sys.argv[1]:
            with open(sys.argv[2], 'r') as file:
                test(prep_test_data(file.read(), output_params[4]))
