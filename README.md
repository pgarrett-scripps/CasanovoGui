## Try it out:
- https://casanovogui.streamlit.app/search

## Install from repo

```bash
git clone https://github.com/pgarrett-scripps/CasanovoGui.git
cd CasanovoGui
pip install .
casanovogui
```

## Install from PyPi

- https://pypi.org/project/casanovogui/
- A new release is automatically uploaded to PyPi upon a new release

```bash
pip install casanovogui
casanovogui
```

# Docker

- https://hub.docker.com/repository/docker/pgarrettscripps/casanovogui/general
- A new release is automatically uploaded to Dockerhub upon a new release

## Dockerhub install

- The docker-compose.yaml file is setup such that it requires a compatible GPU. Using casanovo without a GPU is not recommended due to the amount of time it takes to process files.
- If you want to run casanovogui (via docker) without a GPU, you can modify the docker-compose.yaml file to use the CPU version of the compose file by removing the `deploy` section.

```bash
git clone https://github.com/pgarrett-scripps/CasanovoGui.git
cd CasanovoGui
docker compose up
```

