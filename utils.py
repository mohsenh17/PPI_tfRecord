import numpy as np
import tensorflow as tf
from tfrecord_converter import serialize_embedding, process_and_save_embeddings, serialize_index_pair, process_and_save_protein_pairs

def parse_embedding_function(proto):
    # Describe the features of the serialized data (matches the writing process)
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),                # ID of the embedding (as string)
        'embedding': tf.io.FixedLenFeature([819200], tf.float32),       # Flattened embedding array (adjust size)
        'shape': tf.io.FixedLenFeature([2], tf.int64)               # Shape of the original 2D embedding
    }
    # Parse the input tf.train.Example
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    
    # Reshape the flat embedding back to its original shape
    embedding = tf.reshape(parsed_features['embedding'], parsed_features['shape'])
    
    return parsed_features['id'],parsed_features['shape'], embedding

# Load the TFRecord file using TFRecordDataset
embedding_dataset = tf.data.TFRecordDataset('Allembeddings.tfrecord')

# Apply the parsing function to each record
parsed_dataset = embedding_dataset.map(parse_embedding_function)

# Iterate and print parsed data (ID and 2D embedding matrix)
for id_,shape_, embedding in parsed_dataset:
    print(f"ID: {id_.numpy().decode('utf-8')}")
    print(f"Embedding: {embedding.numpy()}\n")
    print(f"shape: {shape_.numpy()}\n")



# Process and save embeddings
process_and_save_embeddings(
    input_path='dataset/embd/*',
    output_file='Allembeddings.tfrecord',
    embedding_shape=(800, 1024)
)

# Process and save protein pairs
process_and_save_protein_pairs(
    input_file='dataset/pairs/pairs.tsv',
    output_file='Allprotpairs.tfrecord'
)
