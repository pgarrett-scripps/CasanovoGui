FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT casanovogui