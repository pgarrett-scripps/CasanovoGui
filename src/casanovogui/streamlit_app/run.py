import subprocess


def main():
    _ = subprocess.Popen(["streamlit", "run", 'home.py', "--maxUploadSize=2048"])
