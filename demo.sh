mkdir -p "./training/pretrained"
gsutil -m cp -r "gs://kaggle-vbdi-test/nlp-text-summarization/15epochs/checkpoint-65000/*" "./training/pretrained"