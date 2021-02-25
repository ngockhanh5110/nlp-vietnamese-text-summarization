# Summary

Summary task in Vietnamese applies seq2seq model. Thanks to the SOTA Roberta model in Vietnamese, PhoBERT, I made summarization architecture which is trained on Vietnews dataset (reference 1) 

# Demo

1. Step 1: Build docker container

```
docker build -f Dockerfile -t nlp-text-summarization:latest .
```

2. Step 2: Run docker container

```
docker run -p 8501:8501 nlp-text-summarization:latest
```

# Results

The model outperforms the recent research paper on Vietnamese text summarization on the same dataset.

| Attempt | Precision | Recall | F1-Score | F1-Score Fast-Abs (Ref 1) | 
| :---: | :---: | :---: | :---: | :---: |
| Rouge 1 | 0.64 | 0.61 | 0.61 | 0.55 | 
| Rouge 2 | 0.31 | 0.30 | 0.30 | 0.23 |
| Rouge L | 0.42 | 0.41 | 0.40 | 0.38 |

# Reference
1. Nguyen, Van-Hau & Nguyen, Thanh-Chinh & Nguyen, Minh-Tien & Hoai, Nguyen. (2019). VNDS: A Vietnamese Dataset for Summarization. 375-380. 10.1109/NICS48868.2019.9023886. 
2. Rothe, Sascha & Narayan, Shashi & Severyn, Aliaksei. (2020). Leveraging Pre-trained Checkpoints for Sequence Generation Tasks. Transactions of the Association for Computational Linguistics. 8. 264-280. 10.1162/tacl_a_00313. 