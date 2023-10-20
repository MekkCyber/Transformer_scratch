from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel

from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from dataset import BilDataset, causal_mask
from config import get_config, get_weights_file_path
from model import *
from tqdm import tqdm

import os
from pathlib import Path
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
def build_tokenizer(config, ds, lang) : 
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path) : 
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else : 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config) : 
    ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tg']}", split="train")
    # Build Tokenizers
    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tg = build_tokenizer(config, ds_raw, config['lang_tg'])

    train_size = int(0.9*len(ds_raw))
    train_raw, val_raw = random_split(ds_raw,[train_size, len(ds_raw)-train_size])

    train = BilDataset(train_raw, tokenizer_src, tokenizer_tg, config['lang_src'], config['lang_tg'], config['seq_len'])
    val = BilDataset(val_raw, tokenizer_src, tokenizer_tg, config['lang_src'], config['lang_tg'], config['seq_len'])

    max_len_src = 0
    max_len_tg = 0

    for item in ds_raw : 
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tg_ids = tokenizer_tg.encode(item['translation'][config['lang_tg']]).ids
        max_len_src = max(len(src_ids), max_len_src)
        max_len_tg = max(len(tg_ids), max_len_tg)
    print("max length in src language : ",max_len_src)
    print("max length in target language : ",max_len_tg)
    train_dataloader = DataLoader(train, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tg


def get_model(config, src_vocab_size, tg_vocab_size) : 
    model = build_transformer(src_vocab_size, tg_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config) : 
    # 'cuda' if torch.cuda.is_available() else 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tg.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_tg.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(config['num_epochs']) : 
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"epoch {epoch}") 
        global_step = 0
        for batch in batch_iterator : 
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input,encoder_output, encoder_mask, decoder_mask)
            proj = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj.view(-1, tokenizer_tg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #validate(model, val_dataloader, tokenizer_src, tokenizer_tg, config["seq_len"], device, lambda msg : batch_iterator.write(msg), global_step, writer)

            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def greedy_decode(model, source, src_mask, tokenizer_src, tokenizer_tg, max_len, device) : 
    cls = tokenizer_src.token_to_id('[CLS]')
    eos = tokenizer_src.token_to_id('[EOS]')

    encode_output = model.encode(source, src_mask)
    decode_input = torch.tensor([[cls]]).to(device)

    while decode_input.shape[1]<max_len and decode_input[0,-1]!=eos : 
        tg_mask = causal_mask(decode_input.shape[1]).to(device)
        outputs = model.decode(decode_input, encode_output, src_mask, tg_mask)
        prob = model.project(outputs[:,-1])
        # next word 
        _, next_word_idx = torch.max(prob, dim=1)
        decode_input = torch.cat([decode_input, next_word_idx.unsqueeze(0).to(device)], dim=1) 

    return decode_input.squeeze(0)

def validate(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tg_text"][0]
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
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.text.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.text.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.text.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


if __name__ == "__main__" : 
    config = get_config()
    train_model(config)
    
