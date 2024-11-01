import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Constants
BATCH_SIZE = 64
EMBEDDING_SHAPE = (800, 1024)
LEARNING_RATE = 0.001
EMBEDDING_DATASET = "Allembeddings.tfrecord"
PAIR_DATASET = "Allprotpairs.tfrecord"

def create_model(embedding_shape: tuple = EMBEDDING_SHAPE, learning_rate: float = LEARNING_RATE):
    """
    Creates and compiles a neural network model for binary classification.

    Args:
        embedding_shape (tuple): The shape of the input embeddings.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        Model: A compiled Keras model.
    """
    input_features1 = Input(shape=embedding_shape, name="input_1")
    input_features2 = Input(shape=embedding_shape, name="input_2")

    # Process embeddings and concatenate
    out1 = Flatten()(input_features1)
    out2 = Flatten()(input_features2)
    concatenated = Concatenate()([out1, out2])

    # Fully connected layers
    out = Dense(512, activation='relu')(concatenated)
    out = Dropout(rate=0.5)(out)
    out = Dense(64, activation='relu')(out)
    out = Dropout(rate=0.3)(out)
    out = Dense(8, activation='relu')(out)
    out = Dropout(rate=0.3)(out)
    output = Dense(1, activation='sigmoid')(out)

    # Create and compile the model
    model = Model(inputs=[input_features1, input_features2], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def parse_embedding_function(proto: tf.Tensor):
    """
    Parses TFRecord embedding data.

    Args:
        proto (tf.Tensor): The serialized example from the TFRecord file.

    Returns:
        tuple: Parsed ID and embedding as a reshaped tensor.
    """
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'embedding': tf.io.FixedLenFeature([819200], tf.float32),
        'shape': tf.io.FixedLenFeature([2], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    embedding = tf.reshape(parsed_features['embedding'], parsed_features['shape'])
    return parsed_features['id'], embedding

embedding_dataset = tf.data.TFRecordDataset(EMBEDDING_DATASET).map(parse_embedding_function)
embedding_lookup = {id_.numpy().decode('utf-8'): embedding.numpy() for id_, embedding in embedding_dataset}

def parse_index_pair_function(proto: tf.Tensor):
    """
    Parses TFRecord data for index pairs and labels.

    Args:
        proto (tf.Tensor): The serialized example from the TFRecord file.

    Returns:
        tuple: Parsed indices and label.
    """
    feature_description = {
        'index_1': tf.io.FixedLenFeature([], tf.string),
        'index_2': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    return parsed_features['index_1'], parsed_features['index_2'], parsed_features['label']

def map_indices_to_embeddings(index_1: tf.Tensor, index_2: tf.Tensor, label: tf.Tensor):
    """
    Maps indices to their respective embeddings from the lookup dictionary.

    Args:
        index_1 (tf.Tensor): ID of the first embedding.
        index_2 (tf.Tensor): ID of the second embedding.
        label (tf.Tensor): Label for the pair.

    Returns:
        tuple: Embedding tensors for both indices and label.
    """
    embedding_1 = np.array(embedding_lookup.get(index_1.numpy().decode('utf-8'), np.zeros(EMBEDDING_SHAPE)))
    embedding_2 = np.array(embedding_lookup.get(index_2.numpy().decode('utf-8'), np.zeros(EMBEDDING_SHAPE)))
    return embedding_1, embedding_2, label

def tf_map_indices_to_embeddings(index_1: tf.Tensor, index_2: tf.Tensor, label: tf.Tensor):
    """
    Maps indices to embeddings and ensures compatibility with TensorFlow's tf.data.Dataset API.

    Args:
        index_1 (tf.Tensor): ID of the first embedding.
        index_2 (tf.Tensor): ID of the second embedding.
        label (tf.Tensor): Label for the pair.

    Returns:
        tuple: Tensors for both embeddings and label.
    """
    emb1, emb2, label = tf.py_function(
        func=map_indices_to_embeddings,
        inp=[index_1, index_2, label],
        Tout=[tf.float32, tf.float32, tf.int64]
    )
    emb1.set_shape(EMBEDDING_SHAPE)
    emb2.set_shape(EMBEDDING_SHAPE)
    label.set_shape([])
    return emb1, emb2, label

def get_dataset():
    """
    Loads and prepares the dataset for model training.

    Returns:
        tf.data.Dataset: The prepared dataset with embeddings and labels.
    """
    index_pairs_dataset = tf.data.TFRecordDataset(PAIR_DATASET).map(parse_index_pair_function)
    dataset = index_pairs_dataset.map(lambda idx1, idx2, label: tf_map_indices_to_embeddings(idx1, idx2, label))
    dataset = dataset.map(lambda emb1, emb2, label: ({"input_1": emb1, "input_2": emb2}, label))
    dataset = dataset.shuffle(buffer_size=2048).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Instantiate and compile the model
model = create_model()
model.summary()

# Callbacks for training
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)
mc = ModelCheckpoint(
    filepath="out.h5",
    save_weights_only=True,
    monitor='loss',
    mode='min',
    verbose=1,
    save_best_only=True
)

# Training the model
train_dataset = get_dataset()
model.fit(train_dataset, callbacks=[es, mc], epochs=100, verbose=2)

