
# Semantic Search with Sentence-BERT

Welcome to the Semantic Search with Sentence-BERT project! This project focuses on implementing a semantic search engine using the Sentence-BERT model.

## Introduction

Semantic search involves retrieving documents that are semantically similar to a query. In this project, we leverage Sentence-BERT to implement a semantic search engine using a dataset of documents.

## Dataset

For this project, we will use a custom dataset of documents. You can create your own dataset and place it in the `data/documents.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas
- Flask

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/semantic_search_sentence_bert.git
cd semantic_search_sentence_bert

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes documents. Place these files in the data/ directory.
# The data should be in a CSV file with one column: document.

# To train the Sentence-BERT model for semantic search, run the following command:
python scripts/train.py --data_path data/documents.csv

# To evaluate the performance of the model, run:
python scripts/evaluate.py --model_path models/ --data_path data/queries.csv
