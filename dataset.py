import torch 
import torch.nn as nn
from torch.utils.data import Dataset 

class BilDataset(Dataset) : 
    def __init__(self, ds, tokenizer_src, tokenizer_tg, src_lang, tg_lang, seq_len) : 
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tg = tokenizer_tg
        self.src_lang = src_lang
        self.tg_lang = tg_lang
        self.seq_len = seq_len
        self.cls_token = torch.tensor([tokenizer_src.token_to_id('[CLS]')],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self) : 
        return len(self.ds)
    
    def __getitem__(self, index) : 
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tg_text = src_target_pair["translation"][self.tg_lang]
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tg.encode(tg_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0 : 
            raise ValueError("Sentence Too Long")
        
        encoder_input = torch.cat([self.cls_token, torch.tensor(enc_input_tokens, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)])
        decoder_input = torch.cat([self.cls_token, torch.tensor(dec_input_tokens, dtype=torch.int64), torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)])
        label = torch.cat([torch.tensor(dec_input_tokens, dtype=torch.int64), self.cls_token, torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)])

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask" : (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),
            "label" : label,
            "src_text" : src_text,
            "tg_text" : tg_text
        }

def causal_mask(size) : 
    mask = torch.triu(torch.ones([size,size]), diagonal=1).type(torch.int)
    return mask==0