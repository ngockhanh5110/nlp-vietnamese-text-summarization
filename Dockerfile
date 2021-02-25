FROM python:3.7

EXPOSE 8501

WORKDIR /usr/src/app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY demo.sh ./demo.sh

RUN bash demo.sh

COPY jdk.sh ./jdk.sh

RUN bash jdk.sh

COPY . .

CMD streamlit run demo.py