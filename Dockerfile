FROM python:3.7

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD streamlit run demo.py