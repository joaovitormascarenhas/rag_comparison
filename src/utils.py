import pandas as pd
from langchain.schema import Document

def load_dataset(file_path):
    """Carrega o dataset e converte para objetos LangChain Document."""
    df = pd.read_pickle(file_path)

    documents = [
        Document(
            page_content=row['texto'],
            metadata={
                'title': row['title'],
                'summary': row['summary'],
                'source': row['source']
            }
        )
        for _, row in df.iterrows()
    ]
    return documents