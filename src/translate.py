import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from config import get_config, latest_weights_file_path
from model import build_transformer

def translate(sentence_input: str, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    model = build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), 
        config["seq_len"], config['seq_len'], d_model=config['d_model'],
        num_encoder_layers=config.get('num_encoder_layers', 6), 
        num_heads=config.get('num_heads', 8),
        dropout_rate=config.get('dropout_rate', 0.1), 
        d_ff=config.get('d_ff', 2048),
        positional_encoding_type=config.get('positional_encoding_type', 'additive')
    ).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename and Path(model_filename).exists():
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("Error: Model weights not found. Train a model first.", file=sys.stderr)
        return ""

    src_encoded_ids = tokenizer_src.encode(sentence_input).ids
    src_tensor = torch.tensor([tokenizer_src.token_to_id('[SOS]')] + src_encoded_ids + [tokenizer_src.token_to_id('[EOS]')]).unsqueeze(0).to(device)
    src_mask_tensor = (src_tensor != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)

    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask_tensor)
        decoder_input = torch.tensor([[tokenizer_tgt.token_to_id('[SOS]')]], dtype=torch.long, device=device)

        for _ in range(config['seq_len'] - 1):
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask_tensor).to(device)
            output = model.decode(encoder_output, src_mask_tensor, decoder_input, decoder_mask)
            next_token = torch.argmax(output[:, -1], dim=-1)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer_tgt.token_to_id('[EOS]'):
                break

    return tokenizer_tgt.decode(decoder_input.squeeze(0).tolist())

if __name__ == '__main__':
    config = get_config()
    sentence = "Hello World"  # Example input
    translated_text = translate(sentence, config)
    print(f"Translated: {translated_text}")