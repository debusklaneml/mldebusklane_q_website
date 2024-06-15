import pandas as pd
from sentence_transformers import SentenceTransformer


def add_embeddings_to_dataframe(df, column_name):
    # Load the sentence transformer model
    model = SentenceTransformer('thenlper/gte-base')
    
    # Ensure the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Generate embeddings for the specified column.
    # The output is a list of lists (each inner list is an embedding vector for a sentence).
    embeddings = model.encode(df[column_name].to_list(), convert_to_tensor=False, show_progress_bar=True)
    
    # Convert embeddings to a DataFrame.
    embeddings_df = pd.DataFrame(embeddings)
    
    # Rename the columns to indicate they are embedding dimensions.
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
    
    # Concatenate the original DataFrame with the embeddings DataFrame.
    result_df = pd.concat([df, embeddings_df], axis=1)
    
    return result_df
