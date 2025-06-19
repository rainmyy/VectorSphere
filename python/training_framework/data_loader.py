import pandas as pd
from sqlalchemy import create_engine, text
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config


def load_data_from_db(connection_string, schema, tables_limit=10, rows_limit=1000):
    """Loads data from a relational database."""
    engine = create_engine(connection_string)
    with engine.connect() as connection:
        inspector = sqlalchemy.inspect(engine)
        tables = inspector.get_table_names(schema=schema)[:tables_limit]
        
        all_data = []
        for table in tables:
            query = f'SELECT * FROM "{schema}"."{table}" LIMIT {rows_limit}'
            df = pd.read_sql(query, connection)
            # Simple text representation of rows
            for _, row in df.iterrows():
                all_data.append({'text': ', '.join(map(str, row.values))})
                
    return Dataset.from_pandas(pd.DataFrame(all_data))

def get_dataloader(dataset, batch_size, tokenizer_name):
    """Creates a DataLoader for the given dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)