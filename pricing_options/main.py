# pip install streamlit

import subprocess
import sys

def main():
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "Pricing_options/app.py"])

if __name__ == "__main__":
    main()