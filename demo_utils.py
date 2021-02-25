import streamlit as st
import yaml
import os
from general_utils import *

from transformers import RobertaTokenizer, EncoderDecoderModel, AutoTokenizer

from vncorenlp import VnCoreNLP

with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

# # Get the checkpoints from gcp
# os.makedirs(configs['output_dir']+'/pretrained/', exist_ok=True)
# os.system('gsutil -m cp -r "{}/*" "{}"'.format(configs['gcp_pretrained_path'],configs['output_dir']+'/pretrained/'))
model = EncoderDecoderModel.from_pretrained(configs['output_dir']+'/pretrained/')

rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 
