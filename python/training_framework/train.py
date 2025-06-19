import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

import config
from data_loader import get_dataloader, load_data_from_db
from model import get_model_and_tokenizer

def main():
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config.GENERATION_MODEL_NAME)
    model.to(config.DEVICE)

    # Load data
    # This is an example of loading from a database. You can adapt this to load from files.
    if config.DB_CONNECTION_STRING:
        dataset = load_data_from_db(
            config.DB_CONNECTION_STRING, 
            config.DB_SCHEMA
        )
    else:
        # Example of loading from a text file (you need to create this file)
        # with open(os.path.join(config.DATA_PATH, "train.txt"), "r") as f:
        #     texts = f.readlines()
        # dataset = Dataset.from_dict({"text": texts})
        raise ValueError("No training data source specified. Please set DB_CONNECTION_STRING or provide a data file.")


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        warmup_steps=500,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOG_DIR,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset, # Using the same dataset for evaluation as an example
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    final_model_path = os.path.join(config.MODEL_OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()