# Summary

Summary task in Vietnamese applies seq2seq model. Thanks to the SOTA Roberta model in Vietnamese, PhoBERT, I made summarization architecture which is trained on Vietnews dataset (reference 1) 

# Results

The model outperforms the recent research paper on Vietnamese text summarization on the same dataset.

| Attempt | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| Rouge 1 | 0.64 | 0.57 | 0.59 |
| Rouge 2 | 0.29 | 0.26 | 0.27 |
| Rouge L | 0.40 | 0.37 | 0.38 |