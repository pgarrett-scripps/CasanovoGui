FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install && apt-get install -y python3-pip

RUN python3 -m pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT casanovogui