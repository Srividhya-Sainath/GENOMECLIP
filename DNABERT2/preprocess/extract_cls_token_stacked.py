import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import h5py
import os
import numpy as np
import logging
import time

# Initialize the DNABERT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)

# Setup logging
logging.basicConfig(filename='dnabert2_CLS_TOKEN_OVERALL_processing.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def generate_features(file_path, output_dir):
    # Check if the input file exists
    if not os.path.exists(file_path):
        logging.error(f"The file {file_path} does not exist.")
        return

    df = pd.read_csv(file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group by 'Patient_ID' and sort within each group by 'Mutation_Position'
    grouped_df = df.groupby('Patient_ID')

    for patient_id, group in grouped_df:
        start_time = time.time()
        #group = group.sort_values('Mutation_Position')

        hdf5_filename = os.path.join(output_dir, f"{patient_id}.h5")
        if os.path.exists(hdf5_filename):
            logging.info(f"Skipping {patient_id} as file already exists.")
            continue

        logging.info(f"Processing Patient_ID: {patient_id}")
        all_features = []

        try:
            for _, row in group.iterrows():
                alt_sequence = row['Alt_Sequence']
                sequence_length = len(alt_sequence)
                logging.info(f"Patient_ID: {patient_id}, Alt_Sequence length: {sequence_length}")

                inputs = tokenizer(alt_sequence, return_tensors='pt')['input_ids']
                inputs = inputs.to(device)
                
                with torch.no_grad():
                    cls_token = model(inputs)[1].cpu().numpy()
                    cls_token = cls_token.squeeze(0)
                    # Keep the original hidden states without mean pooling
                    all_features.append(cls_token)

            # Stack all features for this patient along the sequence dimension
            stacked_features = np.vstack(all_features)
            final_feature_dimension = stacked_features.shape

            # Save the stacked features in an HDF5 file
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset("feats", data=stacked_features)
                h5f.create_dataset("Patient_ID", data=str(patient_id).encode('utf-8'))

            elapsed_time = time.time() - start_time
            logging.info(f"Processed and saved data for Patient_ID: {patient_id} in {elapsed_time:.2f} seconds")
            logging.info(f"Final feature dimension for Patient_ID: {patient_id} is {final_feature_dimension}")

        except Exception as e:
            logging.error(f"Failed to process Patient_ID: {patient_id} due to error: {e}")

# Paths for the input CSV file and the output directory
csv_file_path = "/mnt/bulk/vidhya/genomeCLIP/DNABERT_2/preprocess/tcga_mutations_controlled.csv"
output_directory = "/mnt/bulk/vidhya/genomeCLIP/tcga_mutations_controlled_cls_token_overall"

# Generate features
generate_features(csv_file_path, output_directory)
