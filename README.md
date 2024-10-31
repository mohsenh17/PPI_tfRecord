# Protein Embedding and Pair Processing

This repository provides tools to process protein embeddings and protein pairs, serializing them into TFRecord format for efficient storage and loading in TensorFlow-based workflows. It’s useful for projects working with protein data in machine learning applications, enabling easy data handling.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Functions](#functions)
- [Usage](#usage)

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Protein-Embedding-Processor.git
cd Protein-Embedding-Processor
pip install tensorflow numpy pandas
```

## Project Overview
This project offers:

    ● Functions to process and save protein embeddings.
    ● Tools to read and serialize protein pairs with labels.
    ● Storage of processed data in TFRecord format for optimized loading and usage in TensorFlow.

## Functions
### 1. `serialize_embedding(id_str, embedding)`
Converts an embedding and its identifier into a TensorFlow-compatible `Example` for TFRecord storage.

- **Parameters**:
  - `id_str` (str): Unique identifier for the embedding (e.g., protein ID).
  - `embedding` (np.ndarray): A 2D NumPy array representing the embedding.
- **Returns**: Serialized TensorFlow `Example` protocol buffer for storage.
### 2. `process_and_save_embeddings(input_path, output_file, embedding_shape=(800, 1024))`

Reads embedding files from a directory, processes them, and saves them in a TFRecord file.

- **Parameters**:
  - `input_path` (str): Directory path to the embedding files.
  - `output_file` (str): Name for the output TFRecord file.
  - `embedding_shape` (tuple): Dimensions of each embedding (default is `(800, 1024)`).

### 3. `serialize_index_pair(index_pair)`

Serializes a pair of protein IDs with an integer label for TFRecord storage.

- **Parameters**:
  - `index_pair` (tuple): Tuple with two string IDs and a label `(index_1, index_2, label)`.
- **Returns**: Serialized TensorFlow `Example` for storage.

### 4. `process_and_save_protein_pairs(input_file, output_file)`

Reads protein pairs from a TSV file, processes them, and stores them in a TFRecord file.

- **Parameters**:
  - `input_file` (str): Path to the TSV file with protein pairs and labels.
  - `output_file` (str): Name for the output TFRecord file.

## Usage

Here's how to use the functions to process embeddings and pairs.

```python
from process_embeddings import process_and_save_embeddings, process_and_save_protein_pairs

# Process and save embeddings
process_and_save_embeddings(
    input_path='../dataset/embd/human/*',
    output_file='Allembeddings.tfrecord',
    embedding_shape=(800, 1024)
)

# Process and save protein pairs
process_and_save_protein_pairs(
    input_file='../dataset/pairs/human_train_balanced.tsv',
    output_file='Allprotpairs.tfrecord'
)
```
This code reads embeddings and protein pairs from specified files and saves them in TFRecord format for seamless integration into TensorFlow workflows.


