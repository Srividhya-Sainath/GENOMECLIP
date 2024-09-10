import pandas as pd
from typing import Dict, List


def preprocess_data(data: pd.DataFrame, text_column: str, chunk_size: int) -> Dict[str, List[str]]:
    grouped_texts = data.groupby('Patient_ID')[text_column].apply(list).to_dict()
    processed_texts = {}

    for patient_id, texts in grouped_texts.items():
        concatenated_chunks = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_with_sep = [text + '[SEP]' for text in chunk]
            concatenated_chunks.append(''.join(chunk_with_sep))
        processed_texts[patient_id] = concatenated_chunks

    return processed_texts

data = pd.read_csv('/Users/vidhyasainath/Desktop/khooj/genomeCLIP/OLD/tcga_mutations_controlled.csv')
processed_texts = preprocess_data(data,'Alt_Sequence',100)
rows = []
for patient_id, sequences in processed_texts.items():
    for sequence in sequences:
        sequence_with_sep = sequence + '[SEP]'
        rows.append((patient_id, sequence_with_sep))

# Creating the DataFrame
df = pd.DataFrame(rows, columns=['Patient_ID', 'Alt_Sequence'])

df.to_csv('/Users/vidhyasainath/Desktop/khooj/genomeCLIP/OLD/tcga_mutations_controlled_batch100_withSEP.csv')
