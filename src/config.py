from pathlib import Path
import google.colab


def get_colab_config():
    return {
        "batch_size": 64,
        "num_epochs": 50,
        "lr": 2e-4,
        "seq_len": 128,
        "d_model": 512,
        "sample_size": 200000,
        "max_val_samples": 1000,
        "use_mixed_precision": True,
        "use_gradient_checkpointing": True,
        "gradient_accumulation_steps": 4,
        "use_lr_scheduler": True,
        "warmup_steps": 700,
        "use_early_stopping": True,
        "patience": 25,
        "datasource": "cfilt/iitb-english-hindi",
        "local_data_path": None,
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "/content/drive/MyDrive/transformer_weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "/content/drive/MyDrive/tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "wandb_project": "transformer-translate-en-hi",
        "is_colab": True,
        "positional_encoding_type": "rope",
        "optimizer_name": "adamw",
        "optimizer_params": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "adam_eps": 1e-9,
            "sgd_momentum": 0.9,
            "rmsprop_alpha": 0.99,
            "weight_decay": 0.01
        },
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "d_ff": 2048,
        "beam_size": 3,
        "length_penalty_alpha": 0.6,
        "validation_freq": 1,
        "save_freq": 1,
        "reduced_vocab": True,
        "num_workers": 2
    }

def get_config():
    return get_colab_config()

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = Path(config['model_folder'])
    weights_files = sorted(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    return str(weights_files[-1])
