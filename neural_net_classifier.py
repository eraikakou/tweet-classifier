import os
import re
import glob

import tensorflow as tf

import numpy as np
from datetime import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from keras_lr_finder import LRFinder

import keras.backend as K
from keras.models import load_model
from keras.backend import tensorflow_backend

from tensorboard.plugins.pr_curve import summary as pr_summary

BASE_DIR = './data'
EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2
BATCH_SIZE= 256

class NNClassifier(object):
    """
        The implementation of a neural net classifier based on Keras
    """
    def __init__(self):
        """ Contructor"""
        self.embeddings_index = {}
        with open(os.path.join(BASE_DIR, 'glove.twitter.27B.100d.txt')) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(self.embeddings_index))
    
    def train(self, train_data, devtest_data):
        train_data.extend(devtest_data)
        # x_train = [np.asarray(x[1].split()) for x in train_data]
        x_train = [x[1] for x in train_data]
        y_train = [x[0] for x in train_data]
        y_train = np.asarray(y_train)
        # finally, vectorize the text samples into a 2D integer tensor
        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, char_level=False)
        self.tokenizer.fit_on_texts(x_train)
        sequences = self.tokenizer.texts_to_sequences(x_train)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = y_train[indices]
        num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-num_validation_samples]
        y_train = labels[:-num_validation_samples]
        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]

        # prepare embedding matrix
        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            # print(i)
            # print(word)
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            # print(embedding_vector)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        print('Training model.')

        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(256, 5, activation='relu')(embedded_sequences)
        x = GlobalMaxPooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        preds = Dense(1, activation='sigmoid')(x)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        plot_model(self.model, to_file='model.png')

        self.model.summary()

        self.model_output_dir = os.path.join(
            'models',
            datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        )

        filepath = os.path.join(
            self.model_output_dir,
            "weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
        )

        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=3,
            verbose=0,
            mode='auto')

        prtensorboard_dir = os.path.join(self.model_output_dir, 'pr_tensorboard_logs')
        lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(x_train.shape[0]/BATCH_SIZE), epochs=10)
        
        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=10,
                       callbacks=[checkpoint, early_stopping, PRTensorBoard(log_dir=prtensorboard_dir), lr_finder],
                       validation_data=(x_val, y_val))

        lr_finder.plot_loss()

    def predict(self, test_data):
        """
            Predict and get accuracy from the provided test data
        """

        x_test = [x[1] for x in test_data]
        y_test = [x[0] for x in test_data]

        sequences = self.tokenizer.texts_to_sequences(x_test)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        self.load_model()
        y_pred = self.model.predict(data)
        y_pred = [str(int(round(y[0]))) for y in y_pred]
        hits = 0
        for i, val in enumerate(y_test):
            if val == y_pred[i]:
                hits += 1
        print("Accuracy:", float(hits) / len(y_pred))

    def load_model(self):
        """Load a model from a provided path"""
        try:
            tensorflow_backend.clear_session()
            self._find_latest_model_path()
            self.model = load_model(self._find_latest_model_path())
            self.graph = tf.get_default_graph()

        except Exception as e:
            print('Could not load model:', str(e))

    def _find_latest_model_path(self):

        latest_model = None
        max_epoch = 0
        files = [
            file_path
            for file_path
            in glob.iglob(os.path.join(self.model_output_dir, 'weights-improvement*'), recursive=True)
        ]
        for file in files:
            file = re.sub(self.model_output_dir, '', file)
            if int(file.split('-')[2]) > max_epoch:
                latest_model = self.model_output_dir + file
        return latest_model


class PRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(PRTensorBoard, self).__init__(*args, **kwargs)

    def set_model(self, model):
        super(PRTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(tag='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(PRTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            # for i in self.validation_data:
                # print(i.shape)
            predictions = self.model.predict(self.validation_data[0])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[1], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
            self.writer.flush()
