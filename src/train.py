import warnings
warnings.filterwarnings("ignore")
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from model import build_transformer
from dataset import BilingualDataset, causal_mask 
from config import get_config, get_weights_file_path, latest_weights_file_path
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import gc
import time
from tqdm import tqdm 
from pathlib import Path
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import Whitespace 
import wandb
import torchmetrics
import sys
import numpy as np

def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device, beam_size=4, length_penalty_alpha=0.6):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    with torch.no_grad():
        encoder_output = model.encode(source, source_mask)
        beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]
        completed = []

        for _ in range(max_len):
            candidates = []

            for seq, score in beams:
                if seq[0, -1].item() == eos_idx:
                    completed.append((seq, score))
                    continue

                decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
                out = model.decode(encoder_output, source_mask, seq, decoder_mask)
                prob = torch.log_softmax(model.project(out[:, -1]), dim=-1)
                topk_prob, topk_idx = torch.topk(prob, beam_size)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, topk_idx[:, i:i+1]], dim=1)
                    new_score = score + topk_prob[:, i].item()
                    candidates.append((new_seq, new_score))

            ordered = sorted(candidates, key=lambda x: x[1]/((x[0].size(1)**length_penalty_alpha) if length_penalty_alpha > 0 else 1.0), reverse=True)
            beams = ordered[:beam_size]

            if len(completed) >= beam_size:
                break

        if completed:
            completed.sort(key=lambda x: x[1]/((x[0].size(1)**length_penalty_alpha) if length_penalty_alpha > 0 else 1.0), reverse=True)
            best_seq = completed[0][0]
        else:
            best_seq = beams[0][0]

        return best_seq.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, config, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80 

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            beam_size = config.get('beam_size', 1)
            length_penalty_alpha = config.get('length_penalty_alpha', 0.0)

            model_out = beam_search_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device, beam_size, length_penalty_alpha)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})
    
    return bleu

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.parent.exists():
        print(f"Creating directory: {tokenizer_path.parent}")
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not tokenizer_path.exists():
        print(f"Building tokenizer for {lang}...")
        vocab_size = 8000 if config.get('reduced_vocab', False) else 16000
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        )
        
        try:
            temp_file = tokenizer_path.parent / "temp_test_file"
            with open(temp_file, 'w') as f:
                f.write("test")
            temp_file.unlink()
        except Exception as e:
            print(f"Error testing write permissions: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Directory permissions: {os.stat(tokenizer_path.parent)}")
            raise
        
        print(f"Training tokenizer for {lang}...")
        
        def batch_iterator(ds, lang, batch_size=1000):
            for i in range(0, len(ds), batch_size):
                batch = ds[i:i+batch_size]
                yield [item['translation'][lang] for item in batch]
        
        tokenizer.train_from_iterator(batch_iterator(ds, lang), trainer=trainer)
        
        try:
            print(f"Saving tokenizer to {tokenizer_path}")
            tokenizer.save(str(tokenizer_path))
        except Exception as e:
            print(f"Error saving tokenizer: {e}")
            print(f"Attempting to save to current directory instead...")
            fallback_path = Path(f"./tokenizer_{lang}.json")
            tokenizer.save(str(fallback_path))
            print(f"Saved tokenizer to {fallback_path}")
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    print(f"\nConfiguration:")
    print(f"Model folder: {config['model_folder']}")
    print(f"Tokenizer path template: {config['tokenizer_file']}")
    print(f"Languages: {config['lang_src']} -> {config['lang_tgt']}")
    
    tokenizer_src_path = Path(config['tokenizer_file'].format(config['lang_src'])).parent
    tokenizer_tgt_path = Path(config['tokenizer_file'].format(config['lang_tgt'])).parent
    model_folder_path = Path(config['model_folder'])
    
    for path in [tokenizer_src_path, tokenizer_tgt_path, model_folder_path]:
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
    
    sample_size = config.get('sample_size', None)
    print(f"Using sample size: {sample_size if sample_size else 'Full dataset'}")
    
    try:
        if 'local_data_path' in config and config['local_data_path']:
            print(f"Loading dataset from {config['local_data_path']}")
            ds_raw = load_from_disk(config['local_data_path'])
        else:
            print(f"Loading dataset from HuggingFace: {config['datasource']}")
            ds_raw = load_dataset(config['datasource'], f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Attempting to load the dataset directly...")
        try:
            ds_raw = load_dataset(config['datasource'], split='train')
            sample = next(iter(ds_raw))
            if 'translation' not in sample or config['lang_src'] not in sample['translation'] or config['lang_tgt'] not in sample['translation']:
                print(f"Dataset does not have the expected format. Sample: {sample}")
                sys.exit(1)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            print("Please check your dataset configuration and internet connection.")
            sys.exit(1)

    if sample_size and sample_size < len(ds_raw):
        print(f"Sampling {sample_size} examples from {len(ds_raw)} total examples")
        np.random.seed(42)
        random_indices = np.random.choice(len(ds_raw), sample_size, replace=False)
        ds_raw = ds_raw.select(random_indices)
    
    print("\nDataset sample:")
    print(next(iter(ds_raw)))
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    max_seq_len = config['seq_len'] - 2
    
    def filter_long_sequences(example):
        src_ids = tokenizer_src.encode(example['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(example['translation'][config['lang_tgt']]).ids
        return len(src_ids) <= max_seq_len and len(tgt_ids) <= max_seq_len

    print("\nFiltering long sequences...")
    original_size = len(ds_raw)
    ds_raw = ds_raw.filter(filter_long_sequences)
    filtered_size = len(ds_raw)
    print(f"Filtered {original_size - filtered_size} sequences ({(original_size - filtered_size) / original_size * 100:.2f}% of data)")

    if filtered_size == 0:
        raise ValueError("All sequences were filtered out. Consider increasing seq_len in config.")

    train_ds_size = int(0.9 * filtered_size)
    val_ds_size = filtered_size - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    max_val_samples = config.get('max_val_samples', 1000)
    if len(val_ds_raw) > max_val_samples:
        val_ds_raw = torch.utils.data.Subset(val_ds_raw, range(max_val_samples))
        print(f"Limited validation set to {max_val_samples} samples")

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    sample_size = min(1000, len(ds_raw))
    sample_indices = torch.randperm(len(ds_raw))[:sample_size]
    
    print(f"\nAnalyzing sequence lengths from {sample_size} samples...")
    for i in sample_indices:
        item = ds_raw[i.item()]
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    num_workers = config.get('num_workers', 2)
    pin_memory_flag = torch.cuda.is_available()
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag
    )
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_flag
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        src_vocab_size=vocab_src_len, 
        tgt_vocab_size=vocab_tgt_len, 
        src_seq_len=config["seq_len"], 
        tgt_seq_len=config['seq_len'], 
        d_model=config['d_model'],
        N=config.get('num_encoder_layers', 6),
        h=config.get('num_heads', 8),
        dropout=config.get('dropout_rate', 0.1),
        d_ff=config.get('d_ff', 2048),
        positional_encoding_type=config.get('positional_encoding_type', 'additive')
    )
    
    model.gradient_checkpointing = config.get('use_gradient_checkpointing', True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.get('use_mixed_precision', True):
        model = model.to(device)
    else:
        model = model.to(device).float()
    
    return model

def train_model(config):
    start_time = time.time()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model size: {model_size:.2f} million parameters")
    wandb.log({'model/size_M': model_size})

    opt_name = config.get('optimizer_name', 'adamw').lower()
    opt_params = config.get('optimizer_params', {})
    lr = config['lr']

    if opt_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=(opt_params.get('adam_beta1', 0.9), opt_params.get('adam_beta2', 0.999)),
            eps=opt_params.get('adam_eps', 1e-8),
            weight_decay=opt_params.get('weight_decay', 0)
        )
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            betas=(opt_params.get('adam_beta1', 0.9), opt_params.get('adam_beta2', 0.98)),
            eps=opt_params.get('adam_eps', 1e-9),
            weight_decay=opt_params.get('weight_decay', 0.01)
        )
    elif opt_name == 'nadam':
        optimizer = torch.optim.NAdam(
            model.parameters(), 
            lr=lr, 
            betas=(opt_params.get('adam_beta1', 0.9), opt_params.get('adam_beta2', 0.999)),
            eps=opt_params.get('adam_eps', 1e-8),
            weight_decay=opt_params.get('weight_decay', 0)
        )
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=opt_params.get('sgd_momentum', 0.0),
            weight_decay=opt_params.get('weight_decay', 0)
        )
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=lr, 
            alpha=opt_params.get('rmsprop_alpha', 0.99),
            eps=opt_params.get('adam_eps', 1e-8),
            weight_decay=opt_params.get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
    
    print(f"Using optimizer: {opt_name} with LR: {lr}")
    print(f"Optimizer params: {opt_params}")

    lr_scheduler = None
    if config.get('use_lr_scheduler', True):
        warmup_steps = config.get('warmup_steps', 4000)
        d_model = config['d_model']
        
        def lr_lambda(step):
            step += 1
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return (float(warmup_steps) ** 0.5) * (float(step) ** -0.5)
            
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    initial_epoch = 0
    global_step = 0
    if config.get('preload'):
        model_filename = config['preload']
        if model_filename == 'latest':
            model_filename = latest_weights_file_path(config)
        else:
            model_filename = get_weights_file_path(config, model_filename)
            
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state.get('epoch', 0) + 1
        optimizer.load_state_dict(state.get('optimizer_state_dict', optimizer.state_dict()))
        global_step = state.get('global_step', 0)
        if lr_scheduler and state.get('scheduler_state_dict'):
            lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        del state
        torch.cuda.empty_cache()
        gc.collect()

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=0.1
    ).to(device)

    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', True) else None
    
    best_bleu = 0.0
    patience = config.get('patience', 3)
    patience_counter = 0
    
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    
    print(f"\nStarting training for {config['num_epochs']} epochs")
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        epoch_loss = 0
        epoch_start_time = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(batch_iterator):
            if scaler:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                with autocast():
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)

                    loss = loss_fn(
                        proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                        label.view(-1)
                    ) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(batch_iterator):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if lr_scheduler is not None:
                        lr_scheduler.step()
            else:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                
                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                    label.view(-1)
                ) / gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(batch_iterator):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if lr_scheduler is not None:
                        lr_scheduler.step()
            
            batch_loss = loss.item() * gradient_accumulation_steps
            epoch_loss += batch_loss
            
            batch_iterator.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}" if lr_scheduler else f"{config['lr']:.6f}"
            })
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                wandb.log({
                    'train/loss': batch_loss,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'global_step': global_step
                })
                global_step += 1
        
        avg_epoch_loss = epoch_loss / len(batch_iterator)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")
        wandb.log({
            'train/epoch_loss': avg_epoch_loss,
            'train/epoch_time': epoch_time,
            'global_step': global_step
        })

        validation_freq = config.get('validation_freq', 1)
        if epoch % validation_freq == 0:
            print(f"Running validation for epoch {epoch}...")
            bleu = run_validation(
                model, 
                val_dataloader, 
                tokenizer_src, 
                tokenizer_tgt, 
                config['seq_len'], 
                device, 
                lambda msg: batch_iterator.write(msg), 
                global_step,
                config
            )
            
            if bleu > best_bleu:
                best_bleu = bleu
                patience_counter = 0
                model_filename = get_weights_file_path(config, f"best")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                    'global_step': global_step,
                    'bleu': bleu
                }, model_filename)
                print(f"New best model saved with BLEU: {bleu:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in BLEU score. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience and config.get('use_early_stopping', True):
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        save_freq = config.get('save_freq', 1)
        if epoch % save_freq == 0:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                'global_step': global_step
            }, model_filename)
            print(f"Model saved as {model_filename}")
            
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Best BLEU score: {best_bleu:.4f}")
    
    best_model_path = get_weights_file_path(config, "best")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    
    return model

if __name__ == '__main__':
    config = get_config()
    for path in [Path(config['model_folder']), 
                 Path(config['tokenizer_file'].format(config['lang_src'])).parent,
                 Path(config['tokenizer_file'].format(config['lang_tgt'])).parent]:
        path.mkdir(parents=True, exist_ok=True)

    wandb.init(project="transformer-en-hi-optimized", config=config)
    trained_model = train_model(config)
    wandb.finish()