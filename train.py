import random
import glob
import pandas as pd
import concurrent.futures
from datasets import *
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 
random.seed(0)

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



train_paths = listPaths('./data/train_tokenized/*')
val_paths = listPaths('./data/val_tokenized/*')
test_paths = listPaths('./data/test_tokenized/*')


train_df = get_dataframe(train_paths)