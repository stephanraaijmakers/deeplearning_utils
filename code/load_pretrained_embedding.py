import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding


# Not tested

def tokenize_pad(docs, max_doc_len):
    #docs = ['a b c','d e f']
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_doc_len, padding='post')
    return (padded_docs, vocab_size)


def load_w2v_embedding(fname, vocab_size, emb_dimension):
    embeddings_index = dict()
    f = open(fname,"r") # some embedding, like glove.6B.100d.txt, or own embedding. Download or create.
    for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, emb_dimension))
    for word, i in t.word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
		    embedding_matrix[i] = embedding_vector
    return embedding_matrix


def train_model(documents, labels, max_document_length, embedding_fname):
    (X, vocab_size)=tokenize_pad(documents, max_document_length) 
    embedding_matrix=load_w2v_embedding(embedding_fname, vocab_size, emb_dimension)
    model = Sequential()
    emb = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=True) # or False
    model.add(emb)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(padded_docs, labels, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))


