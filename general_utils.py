import glob
import concurrent.futures
from transformers import RobertaTokenizerFast,AutoTokenizer
import pandas as pd
from datasets import *
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import yaml
import datasets


with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
encoder_max_length=configs['encoder_max_length']
decoder_max_length=configs['decoder_max_length']

def listPaths(path):
    pathfiles = list()
    for pathfile in glob.glob(path):
        pathfiles.append(pathfile)
    return pathfiles

def read_content(pathfile):
    """
    Input: Path of txt file
    Output: A dictionary has keys 'original' and 'summary'
    """
    with open(pathfile) as f:
        rows  = f.readlines()
        original = ' '.join(''.join(rows[4:]).split('\n'))
        summary = ' '.join(rows[2].split('\n'))
            
    return {'file' : pathfile,
            'original': original, 
            'summary': summary}

def get_dataframe(pathfiles):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = executor.map(read_content, pathfiles)
    
    # Make blank dataframe
    data_df = list()
    for d in data:
        data_df.append(d)
    data_df = pd.DataFrame(data_df)
    data_df.dropna(inplace = True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    return data_df

def process_data_to_model_inputs(batch):                                                               
    # Tokenizer will automatically set [BOS] <text> [EOS]                                               
    inputs = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # mask loss for padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
    return batch  

def get_data_batch(path, batch_size=16, test = False ):
    paths = listPaths(path)
    df = get_dataframe(paths)
    data =  Dataset.from_pandas(df)
    if test:
        return data

    data_batch = data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=["file","original", "summary"],
        )
    data_batch.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
    
    return data_batch

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

# load rouge for validation
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }