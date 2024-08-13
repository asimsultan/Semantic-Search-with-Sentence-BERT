
import os
import torch
import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_device, preprocess_text, DocumentDataset

def main(data_path):
    # Parameters
    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    max_length = 128
    batch_size = 8

    # Load Dataset
    dataset = pd.read_csv(data_path)
    documents = dataset['document'].tolist()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Preprocess Data
    document_encodings = preprocess_text(tokenizer, documents, max_length)

    # DataLoader
    document_dataset = DocumentDataset(document_encodings)
    document_loader = torch.utils.data.DataLoader(document_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = get_device()
    model = BertModel.from_pretrained(model_name)
    model.to(device)

    # Get Document Embeddings
    document_embeddings = []

    for batch in document_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            document_embeddings.append(embeddings.cpu().numpy())

    document_embeddings = np.vstack(document_embeddings)

    # Save Embeddings
    np.save('./models/document_embeddings.npy', document_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing documents')
    args = parser.parse_args()
    main(args.data_path)
