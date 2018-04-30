#Authors: @eldinsahbaz @rohitkulkarni93
import pandas as pd, numpy as np, tensorflow as tf, re, time, sys, contractions, _pickle as pickle, os, nltk, random
from numpy import newaxis
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.python.layers.core import Dense
from nltk.corpus import stopwords
from multiprocessing import Pool
from collections import Counter
from pprint import pprint
from keras.models import Model
from keras.layers import * #Input, CuDNNLSTM, LSTM, Dense, Embedding, TimeDistributed, GRU, CuDNNGRU, Bidirectional
from keras.optimizers import * #RMSprop
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import *

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
        data = [tuple(x) for x in pd.read_csv(path)[['Summary', 'Text']].values.tolist()]
        pool = Pool()

        cleaned = pool.starmap(filter_symbols, data)

        pool.close()
        pool.join()

        cleaned = list(filter(lambda y: y, cleaned))

        with open(target, 'wb') as file: pickle.dump(cleaned, file)

    return cleaned

def create_embeddings(data, cutoff, embedding_map_target, embedding_summary_target, embedding_review_target):
    DNS = {'forward':{'<PAD>':0, '<UNK>':1, '<EOS>':2, '<GO>':3}, 'backward':{0:'<PAD>', 1:'<UNK>', 2:'<EOS>', 3:'<GO>'}}
    stop_words = set(stopwords.words('english'))
    words = list()
    embedded_summaries, embedded_reviews = list(), list()
    plaintext_summaries, plaintext_reviews = list(), list()
    
    #create mapping for word -> int and for int -> word
    for summary, review in data:
        plaintext_summaries.append(sum(summary, []))
        plaintext_reviews.append(sum(review, []))
        words.extend(sum(summary, []))
        words.extend(sum(review, []))

    word_frequencies = [x for x in sorted(Counter(words).items(), key=lambda x: x[1], reverse=True) if (x[1] >= cutoff)][:1000] #(x[0] not in stop_words)

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
        if len(temp) < max_summary_length: temp[len(temp):len(temp)] = [DNS['forward']['<PAD>']]*(max_summary_length-len(temp))
        truncated_summaries.append(temp)

    for review in embedded_reviews:
        temp = review[:max_review_length]
        temp[-1] = DNS['forward']['<EOS>']
        temp = list(reversed(temp))
        if len(temp) < max_review_length: temp[0:0] = [DNS['forward']['<PAD>']]*(max_review_length-len(temp))
        truncated_reviews.append(temp)

    cleaned_truncated_summaries, cleaned_truncated_reviews = list(), list()
    for summary, review in list(zip(truncated_summaries, truncated_reviews)):
        summary_count, review_count = Counter(summary), Counter(review)

        if ((summary_count[DNS['forward']['<UNK>']] <= unk_summary_limit) and (review_count[DNS['forward']['<UNK>']] <= unk_review_limit) and (len(summary) >= min_length) and (len(review) >= min_length)):
            cleaned_truncated_summaries.append(summary)
            cleaned_truncated_reviews.append(review)

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

    with open("max_summary_length", 'wb') as file:
        pickle.dump(max_summary_length, file)

    with open("max_review_length", 'wb') as file:
        pickle.dump(max_review_length, file)

    with open("min_length", 'wb') as file:
        pickle.dump(min_length, file)

    with open("unk_summary_limit", 'wb') as file:
        pickle.dump(unk_summary_limit, file)

    with open("unk_review_limit", 'wb') as file:
        pickle.dump(unk_review_limit, file)

    return (DNS, cleaned_truncated_summaries, cleaned_truncated_reviews, max_summary_length, max_review_length, min_length, unk_review_limit, unk_summary_limit)

def build_model(num_encoder_tokens, vocab_length):
    def build_encoder(input_layer, embedding, recurrent_layers):
        previous_layer = input_layer
        previous_layer = embedding(previous_layer)
        for i in range(len(recurrent_layers)-1):
            previous_layer = recurrent_layers[i](previous_layer)
        return recurrent_layers[-1](previous_layer)

    def build_decoder(input_layer, embedding, initial_state, recurrent_layers, fully_connected):
        previous_layer = input_layer
        previous_layer = embedding(previous_layer)
        for i in range(len(recurrent_layers)): previous_layer = recurrent_layers[i](previous_layer, initial_state=initial_state)
        return fully_connected(previous_layer)

    sparse_cross_entropy = lambda ground_truth, predicted: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ground_truth,logits=predicted))
    state_size, num_encoders_layers, num_decoder_layers = 128, 2, 2
    encoder_layers, decoder_layers = list(), list()

    encoder_input = Input(shape=(None,), name='encoder_input')
    encoder_embedding = Embedding(input_dim=vocab_length, output_dim=128, name='encoder_embedding')
    for i in range(num_encoders_layers-1): encoder_layers.append(CuDNNGRU(state_size, name='encoder_gru{0}'.format(str(i)), return_sequences=True))
    encoder_layers.append(CuDNNGRU(state_size, name='encoder_gru{0}'.format(str(num_encoders_layers)), return_sequences=False))
    encoder_output = build_encoder(encoder_input, encoder_embedding, encoder_layers)

    decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')
    decoder_input = Input(shape=(None,), name='decoder_input')
    decoder_embedding = Embedding(input_dim=vocab_length, output_dim=num_encoder_tokens, name='decoder_embedding')
    for i in range(num_decoder_layers): decoder_layers.append(CuDNNGRU(state_size, name='decoder_gru{0}'.format(str(i)), return_sequences=True))
    decoder_FC = Dense(vocab_length, activation='linear', name='decoder_output')
    encoder_decoder_output = build_decoder(decoder_input, decoder_embedding, encoder_output, decoder_layers, decoder_FC)
    decoder_output = build_decoder(decoder_input, decoder_embedding, decoder_initial_state, decoder_layers, decoder_FC)

    model_train = Model(inputs=[encoder_input, decoder_input], outputs=[encoder_decoder_output])
    model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])
    model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])

    decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
    model_train.compile(optimizer=RMSprop(lr=1e-3), loss=sparse_cross_entropy, target_tensors=[decoder_target])

    print(model_train.summary())
    return (model_train, model_encoder, model_decoder)


def train_and_save(model, model_encoder, model_decoder, encoder_input_data, decoder_input_data, decoder_output_data, modelDir, modelFileName):

    batch_size = 64  # Batch size for training.
    epochs = 1000  # Number of epochs to train for.
    num_samples = 20000  # Number of samples to train on.
    
    xTrain, yTrain = dict(), dict()
    xTrain['encoder_input'] = encoder_input_data[:num_samples]
    xTrain['decoder_input'] = decoder_input_data[:num_samples]
    yTrain['decoder_output'] = decoder_output_data[:num_samples]
    
    #print('Encoder shape:', np.shape(xTrain['encoder_input']))

    path_checkpoint = 's2s_mode.checkpoint'
    my_callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1), ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)]
    
    #print('Decoder input:', np.shape(xTrain['decoder_input']))

    model.fit(x=xTrain, y=yTrain, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=my_callbacks) #0.05

    model_encoder.save_weights(modelDir + 'weights_encoder_' + modelFileName)
    with open(modelDir + 'encoder_' + modelFileName, 'w') as file:
        file.write(model_encoder.to_json())

    model_decoder.save_weights(modelDir + 'weights_decoder_' + modelFileName)
    with open(modelDir + 'decoder_' + modelFileName, 'w') as file:
        file.write(model_decoder.to_json())

def prep_test_data(text, DNS, max_length):
    temp_text = list()
    cleaned_test = filter_symbols_test(text)
    temp_text.append(DNS['forward']['<GO>'])

    for word in nltk.word_tokenize(text):
        try: temp_text.append(DNS['forward'][word])
        except: temp_text.append(DNS['forward']['<UNK>'])

    temp_text.append(DNS['forward']['<EOS>'])

    temp_text = temp_text[:max_length]
    temp_text[-1] = DNS['forward']['<EOS>']
    temp_text = list(reversed(temp_text))
    if len(temp_text) < max_length: temp_text[0:0] = [DNS['forward']['<PAD>']]*(max_length-len(temp_text))
    
    return temp_text

def test(original, text, max_tokens, DNS, modelDir, modelFileName):
    with open(modelDir + 'encoder_' + modelFileName, 'r') as file:
        encoder = model_from_json(file.read())
        encoder.load_weights(modelDir + 'weights_encoder_' + modelFileName)

    with open(modelDir + 'decoder_' + modelFileName, 'r') as file:
        decoder = model_from_json(file.read())
        decoder.load_weights(modelDir + 'weights_decoder_' + modelFileName)

    summary = ''
    generated_summary_length = 0

    encoded_cell_state = encoder.predict(text)
    token_int = DNS['forward']['<GO>']
    #print(token_int)
    decoder_input_data = np.zeros(shape=(1, 10), dtype=np.int)
    while token_int != DNS['forward']['<EOS>'] and generated_summary_length < max_summary_length:
        decoder_input_data[0, generated_summary_length] = token_int
        x_data = dict()
        x_data['decoder_initial_state'] = encoded_cell_state
        x_data['decoder_input'] = decoder_input_data
        next_token = decoder.predict(x_data)
        token_onehot_encoded = next_token[0, generated_summary_length, :]
        token_int = np.argmax(token_onehot_encoded)
        next_word = DNS['backward'][token_int]
        summary += " " + next_word
        generated_summary_length = generated_summary_length + 1

    # Print the input-text.
    print("Input text:")
    print(original)
    print()

    # Print the translated output-text.
    print("Summary Text:")
    print(summary)
    print()

def prepare_decoder_data(embedded_summaries):
    decoder_target_data = np.zeros((len(embedded_summaries), max_summary_length), dtype='float32')

    # Shift decoder data ahead by 1 step and remove start character.
    for i, target_text in enumerate(embedded_summaries):
        for t, word_as_number in enumerate(target_text):
            if t > 0:
                decoder_target_data[i, t - 1] = word_as_number
    return decoder_target_data

reviews_file = 'Reviews.csv'
cleaned_reviews_file = 'cleaned_2.txt'
word_number_mapping_file = "Embedding_Map.txt"
processed_summaries_file = "Embedded_Summary.txt"
processed_reviews_file = "Embedded_Review.txt"
modelDir = './'
modelFileName = 's2s_model.h5'

with tf.device('/device:GPU:0'):
    if (__name__ == '__main__') and (len(sys.argv) > 1):
        if 'train' == sys.argv[1]:

            cutoff = 15
            (DNS, embedded_summaries, embedded_reviews, max_summary_length, max_review_length, min_length, unk_review_limit, unk_summary_limit) = create_embeddings(clean_data(reviews_file, cleaned_reviews_file), cutoff, word_number_mapping_file, processed_summaries_file, processed_reviews_file)
            
            encoder_input_data = np.array(embedded_reviews)
            decoder_input_data = np.array(embedded_summaries)
            decoder_target_data = prepare_decoder_data(embedded_summaries)
            #print("encoder", encoder_input_data[0])
            #print("decoder in", decoder_input_data[0])
            #print("decoder out", decoder_target_data[0])

            model, model_encoder, model_decoder = build_model(max_review_length, len(DNS['forward']))
            train_and_save(model, model_encoder, model_decoder, encoder_input_data, decoder_input_data, decoder_target_data, modelDir, modelFileName)
            
            with open(sys.argv[2], 'r') as file:
                sentences = file.read().split("\n\n")
                for zero in sentences:
                    a = np.array(prep_test_data(zero, DNS, max_review_length))
                    b = a[newaxis,...]
                    #print(np.shape(b))
                    test(zero, b, max_summary_length, DNS, modelDir, modelFileName)

        elif 'test' == sys.argv[1]:
            with open(word_number_mapping_file, 'rb') as file:
                DNS = pickle.loads(file.read())

            with open(processed_summaries_file, 'rb') as file:
                embedded_summaries = pickle.loads(file.read())

            with open(processed_reviews_file, 'rb') as file:
                embedded_reviews = pickle.loads(file.read())

            with open("max_summary_length", 'rb') as file:
                max_summary_length = pickle.loads(file.read())

            with open("max_review_length", 'rb') as file:
                max_review_length = pickle.loads(file.read())

            with open("min_length", 'rb') as file:
                min_length = pickle.loads(file.read())

            with open("unk_summary_limit", 'rb') as file:
                unk_summary_limit = pickle.loads(file.read())

            with open("unk_review_limit", 'rb') as file:
                unk_review_limit = pickle.loads(file.read())

            with open(sys.argv[2], 'r') as file:
                sentences = file.read().split("\n\n")
                for zero in sentences:
                    a = np.array(prep_test_data(zero, DNS, max_review_length))
                    b = a[newaxis,...]
                    #print(np.shape(b))
                    test(zero, b, max_summary_length, DNS, modelDir, modelFileName)
