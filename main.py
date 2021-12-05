import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score

import spacy
import argparse
import time
from datetime import timedelta

from models.transformer import Encoder, Decoder, Seq2Seq

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,required=True)
    parser.add_argument('--lr', type=float,default=0.0005)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch_size',type=int,default=128)
    args = parser.parse_args()
    return args

###############
# Train model
###############
def train(model, iterator, optimizer, criterion, clip):
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


################
# Validate model
################

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def train_and_validate(model, iterator, optimizer, criterion, epochs, clip):
    print("Start Training...")
    model.train()

    for epoch in range(epochs):
        
        train_loss = train(model, iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, iterator, criterion)

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}')

    # save the last model
    torch.save(model.state_dict(), 'checkpoints/trained_model.pt')
    print("Completed training!")

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):

    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)*100

if __name__=="__main__":
    args = args_parser()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create tokenizers
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text): return [tok.text for tok in spacy_de.tokenizer(text)]
    def tokenize_en(text): return [tok.text for tok in spacy_en.tokenizer(text)]

    # Prepare Datasets
    print("Prepare Datasets...")
    SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

    TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)
    
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))
    # Build Vocab
    print("Buid Vocab...")
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    # Iterator
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
     batch_size = args.batch_size,
     device = device)

    # Init model
    # hyper params
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab) 
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    encoder = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

    decoder = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    start_time = time.time()

    if args.mode=="train":
        train_and_validate(model, train_iterator, optimizer, criterion, args.epochs, 1)
    elif args.mode=="test":
        model.load_state_dict(torch.load('checkpoints/trained_model.pt'))
        test_loss = evaluate(model, test_iterator, criterion)
        print(f'Test Loss: {test_loss:.3f}')
        
        # Translate test examples
        example_idx = 8

        src = vars(test_data.examples[example_idx])['src']
        trg = vars(test_data.examples[example_idx])['trg']

        print(f'Source sentence: {src}')
        print(f'True sentence: {trg}')
        
        translation, attention = translate_sentence(src, SRC, TRG, model, device)
        print(f'Predicted sentence: {translation}')

        # Calculate BLEU score
        bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
        print(f'BLEU score: {bleu_score:.2f}')

    else:
        assert("mode should be train or test")

    end_time = time.time()
    print(f'Elapsed time: {str(timedelta(seconds=(end_time-start_time)))}')


    
