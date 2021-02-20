# Getting Vietnews dataset
mkdir -p data
wget 'https://github.com/ThanhChinhBK/vietnews/archive/master.zip' -O ./data/vietnews.zip
unzip './data/vietnews.zip' && rm './data/vietnews.zip'

mv './vietnews-master/data/test_tokenized' './data/test_tokenized'
mv './vietnews-master/data/train_tokenized' './data/train_tokenized'
mv './vietnews-master/data/val_tokenized' './data/val_tokenized'
rm -r './vietnews-master'

mkdir -p training

# Install the requirements.txt
pip install -r requirements.txt