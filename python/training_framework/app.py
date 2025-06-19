import os
import argparse
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import config

app = Flask(__name__)

# Load the fine-tuned model
model_path = os.path.join(config.MODEL_OUTPUT_DIR, "final_model")

# Check if a fine-tuned model exists, otherwise use the base model
if os.path.exists(model_path):
    print(f"Loading fine-tuned model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
else:
    print(f"Fine-tuned model not found. Loading base model {config.GENERATION_MODEL_NAME}")
    from model import get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer(config.GENERATION_MODEL_NAME)

model.to(config.DEVICE)

# Create a text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if config.DEVICE == "cuda" else -1
)

@app.route('/api/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    generated_text = text_generator(prompt, max_length=max_length, num_return_sequences=1)
    return jsonify({'generated_text': generated_text[0]['generated_text']})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the LLM service.")
    parser.add_argument("--port", type=int, default=5001, help="Service port")
    args = parser.parse_args()

    app.run(host="0.0.0.0", port=args.port)