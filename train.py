import transformers
from transformers import RobertaTokenizerFast,AutoTokenizer
from transformers import EncoderDecoderModel

from general_utils import *

import yaml

with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

train_data_batch = get_data_batch(path='./data/train_tokenized/*', batch_size=configs['batch_size'])
val_data_batch = get_data_batch(path='./data/val_tokenized/*', batch_size=configs['batch_size'])

roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/phobert-base", "vinai/phobert-base", tie_encoder_decoder=True)
# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = configs['max_length']
roberta_shared.config.early_stopping = configs['early_stopping']
roberta_shared.config.no_repeat_ngram_size = configs['no_repeat_ngram_size']
roberta_shared.config.length_penalty = configs['length_penalty']
roberta_shared.config.num_beams = configs['num_beams']
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size  

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir= configs['output_dir'],
    per_device_train_batch_size=configs['batch_size'],
    per_device_eval_batch_size=configs['batch_size'],
    predict_with_generate=configs['predict_with_generate'],
    do_train=configs['do_train'],
    do_eval=configs['do_eval'],
    logging_steps=configs['logging_steps'],  
    save_steps=configs['save_steps'], 
    eval_steps=configs['eval_steps'], 
    warmup_steps=configs['warmup_steps'],  
    num_train_epochs=configs['num_train_epochs'], 
    overwrite_output_dir=configs['overwrite_output_dir'],
    save_total_limit=configs['save_total_limit'],
    fp16=configs['fp16'], 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data_batch,
    eval_dataset=val_data_batch,
)
trainer.train()