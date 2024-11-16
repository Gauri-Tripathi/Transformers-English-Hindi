import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_padding_len = self.seq_len - len(enc_input_tokens) - 2
        dec_padding_len = self.seq_len - len(dec_input_tokens) - 1

        if enc_padding_len < 0 or dec_padding_len < 0:
            raise ValueError(f"Sentence is too long for seq_len {self.seq_len}. Source: {len(enc_input_tokens)}, Target: {len(dec_input_tokens)}")

        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            torch.tensor(enc_input_tokens, dtype=torch.long),
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.full((enc_padding_len,), self.pad_token_id, dtype=torch.long),
        ])

        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            torch.tensor(dec_input_tokens, dtype=torch.long),
            torch.full((dec_padding_len,), self.pad_token_id, dtype=torch.long),
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.long),
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.full((dec_padding_len,), self.pad_token_id, dtype=torch.long),
        ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size), dtype=torch.bool), diagonal=1)
    return ~mask