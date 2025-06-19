from transformers import AutoModelForCausalLM, AutoTokenizer

import config

def get_model_and_tokenizer(model_name):
    """Loads a pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad_token_id to eos_token_id if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    return model, tokenizer