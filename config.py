from pathlib import Path

def get_config() : 
    return {
        "batch_size" : 8,
        "num_epochs" : 10,
        "lr" : 1e-4,
        "d_model": 512,
        "seq_len" : 320,
        "lang_src" : "en",
        "lang_tg" : "it",
        "datasource": 'opus_books',
        "model_folder": "weights",
        "model_basename": "tmodel",
        "model_filename" : "transformer",
        "preload": None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name": "runs"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)