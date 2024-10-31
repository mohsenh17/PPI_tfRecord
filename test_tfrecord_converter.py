import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, mock_open
from io import BytesIO
from tfrecord_converter import serialize_embedding, process_and_save_embeddings, serialize_index_pair, process_and_save_protein_pairs

class TestEmbeddingsAndPairs(unittest.TestCase):

    def test_serialize_embedding(self):
        id_str = 'protein_1'
        embedding = np.random.rand(800, 1024)
        
        serialized = serialize_embedding(id_str, embedding)
        
        example = tf.train.Example()
        example.ParseFromString(serialized)
        
        # Verify the serialized content
        self.assertEqual(example.features.feature['id'].bytes_list.value[0].decode('utf-8'), id_str)
        self.assertTrue(np.allclose(
            np.array(example.features.feature['embedding'].float_list.value).reshape(800, 1024),
            np.array(embedding)
        ))
        self.assertEqual(
            list(example.features.feature['shape'].int64_list.value),
            list(embedding.shape)
        )

    @patch('glob.glob')
    @patch('builtins.open', new_callable=mock_open, read_data='id: 1.0 2.0 3.0\n')
    def test_process_and_save_embeddings(self, mock_file, mock_glob):
        mock_glob.return_value = ['path/to/embedding_file.txt']
        output_file = 'test_embeddings.tfrecord'
        
        # Process embeddings
        process_and_save_embeddings('path/to/embedding_file.txt', output_file, (1,3))
        dataset = tf.data.TFRecordDataset(output_file)

        # Check if the file was written
        for record in dataset:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # Check that the ID is processed correctly
            self.assertEqual(example.features.feature['id'].bytes_list.value[0].decode('utf-8'), 'embedding_file')
            # Check embedding values
            embedding_values = np.array(example.features.feature['embedding'].float_list.value).reshape(1,3)
            self.assertTrue(np.allclose(embedding_values[0, :3], [1.0, 2.0, 3.0]), "{}, {}".format([1.0, 2.0, 3.0], embedding_values[0, :3]))  # Check first values



    def test_serialize_index_pair(self):
        index_pair = ('protein_1', 'protein_2', 1)
        
        serialized = serialize_index_pair(index_pair)
        
        example = tf.train.Example()
        example.ParseFromString(serialized)

        # Verify the serialized content
        self.assertEqual(example.features.feature['index_1'].bytes_list.value[0].decode('utf-8'), 'protein_1')
        self.assertEqual(example.features.feature['index_2'].bytes_list.value[0].decode('utf-8'), 'protein_2')
        self.assertEqual(example.features.feature['label'].int64_list.value[0], 1)

    @patch('builtins.open', new_callable=mock_open, read_data='protein_1\tprotein_2\t1\nprotein_3\tprotein_4\t0\n')
    def test_process_and_save_protein_pairs(self, mock_file):
        output_file = 'test_protein_pairs.tfrecord'

        # Process protein pairs
        process_and_save_protein_pairs('input_file.tsv', output_file)
        dataset = tf.data.TFRecordDataset(output_file)

        # Verify the number of records written
        self.assertEqual(sum(1 for _ in dataset), 2)

        # Verify the content of the records
        for idx, record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            if idx == 0:
                self.assertEqual(example.features.feature['index_1'].bytes_list.value[0].decode('utf-8'), 'protein_1')
                self.assertEqual(example.features.feature['index_2'].bytes_list.value[0].decode('utf-8'), 'protein_2')
                self.assertEqual(example.features.feature['label'].int64_list.value[0], 1)
            elif idx == 1:
                self.assertEqual(example.features.feature['index_1'].bytes_list.value[0].decode('utf-8'), 'protein_3')
                self.assertEqual(example.features.feature['index_2'].bytes_list.value[0].decode('utf-8'), 'protein_4')
                self.assertEqual(example.features.feature['label'].int64_list.value[0], 0)


if __name__ == '__main__':
    unittest.main()
