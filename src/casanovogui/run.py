import subprocess
import os
import signal
import sys


def main():
    # Resolve the path to home.py
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "home.py"))

    # Run the Streamlit app with specified arguments
    process = subprocess.Popen(
        ["streamlit", "run", script_path,
            f"--server.maxUploadSize={int(1024*10)}"],
    )

    def signal_handler(_, __):
        print('Terminating Streamlit process...')
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    process.wait()


if __name__ == "__main__":
    main()
