import tensorflow as tf
import numpy as np
import pandas as pd
import glob

def serialize_embedding(id_str: str, embedding: np.ndarray):
    """
    Serializes an embedding and its ID into a TensorFlow `Example` for TFRecord storage.

    Args:
        id_str (str): A unique identifier for the embedding which is protein id.
        embedding (np.ndarray): A 2D NumPy array representing the embedding.

    Returns:
        bytes: A serialized `Example` protocol buffer containing the ID, flattened embedding, 
               and the shape of the original embedding.

    This function flattens the embedding array and stores it along with its ID and shape in a 
    TensorFlow `Example`. The serialized example is suitable for storage in a TFRecord file.
    """
    feature = {
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_str.encode('utf-8')])),
        'embedding': tf.train.Feature(float_list=tf.train.FloatList(value=embedding.flatten())),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=embedding.shape))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def process_and_save_embeddings(input_path: str, output_file: str, embedding_shape: tuple = (800, 1024)):
    """
    Processes embedding files from the specified directory, stores them in a dictionary, 
    and saves them in TFRecord format.

    Args:
        input_path (str): Path pattern to the directory containing embedding files.
        output_file (str): The filename for the output TFRecord file.
        embedding_shape (tuple): Shape of each embedding (default is (800, 1024)).

    """
    embeddings = {}
    
    # Read embedding files and store embeddings
    for address in glob.glob(input_path):
        with open(address, 'r') as embdFile:
            embd_value = np.zeros(embedding_shape)
            for index, line in enumerate(embdFile):
                line = line.strip().split(':')[1].split()
                embd_value[index] = np.array([float(x) for x in line])
            protId = address.split('/')[-1].split('.')[0]
            embeddings[protId] = embd_value

            # Print progress every 1000 embeddings
            if len(embeddings) % 1000 == 0:
                print(f"Processed {len(embeddings)} embeddings")

    # Save embeddings to TFRecord file
    with tf.io.TFRecordWriter(output_file) as writer:
        for id_str, embedding in embeddings.items():
            example = serialize_embedding(id_str, embedding)
            writer.write(example)
