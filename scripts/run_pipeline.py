import subprocess
import sys

def run_command(command):
    print(f'Running: {command}')
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f'Error running: {command}')
        sys.exit(1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol, e.g. SPY or AAPL')
    args = parser.parse_args()

    ticker = args.ticker.upper()

    run_command(f'python scripts/prepare_data.py --ticker {ticker}')
    run_command(f'python scripts/train.py --ticker {ticker}')
    run_command(f'python scripts/evaluate.py --ticker {ticker}')
