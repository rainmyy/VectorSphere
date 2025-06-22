# Large Language Model Training Framework

This framework provides a structured way to fine-tune large language models on your own data.

## Directory Structure

```
python/
└── training_framework/
    ├── app.py              # Flask app to serve the trained model
    ├── c.py           # All configurations and hyperparameters
    ├── data_loader.py      # Scripts for loading and preprocessing data
    ├── model.py            # Model definition
    ├── train.py            # Main training script
    └── README.md           # This file
```

## How to Use

### 1. Installation

Make sure you have all the required packages installed from the parent directory's `requirements.txt`.

```bash
pip install -r ../requirements.txt
```

### 2. Configuration

Edit `c.py` to set your desired parameters, such as model names, data paths, and training hyperparameters.

### 3. Data Preparation

Place your training data in the `data/` directory (you may need to create it). The `data_loader.py` is currently set up to load from a database. You can modify it to load from text files or other formats.

For database training, set the `DB_CONNECTION_STRING` in `c.py`.

### 4. Training

Run the training script:

```bash
python train.py
```

The script will load the data, fine-tune the model, and save the final version to the directory specified by `MODEL_OUTPUT_DIR` in the c.

### 5. Serving the Model

Once training is complete, you can serve the model using the Flask application:

```bash
python app.py --port 5001
```

### 6. Sending Requests

You can send requests to the running service to get text generations:

```bash
curl -X POST http://localhost:5001/api/generate -H "Content-Type: application/json" -d '{"prompt": "Hello, who are you?"}'
```