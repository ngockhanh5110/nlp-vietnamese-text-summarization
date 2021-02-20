import yaml
import os
from general_utils import *

with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

    
def check_empty_folder(path):
    paths = os.listdir(path)
    if len(paths) == 0:
        return True
    else:
        return False

# Get the checkpoints from gcp
if check_empty_folder(r'./training'):
    os.system('gsutil -m cp -r "{}/*" "{}"'.format(configs['gcp_path'],configs['output_dir']))


test_data_batch = get_data_batch(path='./data/test_tokenized/*', batch_size=configs['batch_size'])
