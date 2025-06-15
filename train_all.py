import subprocess

tickers = ['QQQ', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'JPM', 'XLF', 'GLD', 'SLV', 'USO']

for ticker in tickers:
    print(f"\n=== Running pipeline for {ticker} ===\n")
    try:
        subprocess.run(['python', 'scripts/prepare_data.py', '--ticker', ticker], check=True)
        subprocess.run(['python', 'scripts/train.py', '--ticker', ticker], check=True)
        subprocess.run(['python', 'scripts/evaluate.py', '--ticker', ticker], check=True)

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed for {ticker}: {e}\n")
