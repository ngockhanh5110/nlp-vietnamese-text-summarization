# Getting Vietnews dataset
mkdir -p data
wget 'https://github.com/ThanhChinhBK/vietnews/archive/master.zip' -O ./data/vietnews.zip
unzip './data/vietnews.zip' && rm './data/vietnews.zip'

# Install the requirements.txt
pip install -r requirements.txt