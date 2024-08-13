
import os
import torch
import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_device, preprocess_text

def main(model_path, data_path):
    # Load Model and Tokenizer
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Load Embeddings
    document_embeddings = np.load('./models/document_embeddings.npy')

    # Device
    device = get_device()
    model.to(device)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    queries = dataset['query'].tolist()

    # Preprocess Queries
    query_encodings = preprocess_text(tokenizer, queries, max_length=128)

    # Get Query Embeddings
    query_embeddings = []

    for query in query_encodings:
        input_ids = query['input_ids'].unsqueeze(0).to(device)
        attention_mask = query['attention_mask'].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state.mean(dim=1)
            query_embeddings.append(embedding.cpu().numpy())

    query_embeddings = np.vstack(query_embeddings)

    # Calculate Similarity
    similarities = cosine_similarity(query_embeddings, document_embeddings)

    # Print Results
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        print("Most Similar Document Indexes:", similarities[i].argsort()[-3:][::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing queries')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
