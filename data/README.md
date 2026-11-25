# Data

No data is included in this repository. Place the files as follows (paths are relative to the repository root).

## LOBSTER sample files (required)

Free 10-level samples from <https://lobsterdata.com/info/DataSamples.php> (a short request form is required there).
Download the five sample zips for 2012-06-21 (AAPL, AMZN, GOOG, INTC, MSFT) and unzip each into its own folder:

```
data/lobster/AAPL/AAPL_2012-06-21_34200000_57600000_message_10.csv
data/lobster/AAPL/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
data/lobster/AMZN/...   data/lobster/GOOG/...   data/lobster/INTC/...   data/lobster/MSFT/...
```

The code finds the files by the words `message` and `orderbook` in their names. Please check LOBSTER's terms before redistributing the data.

## FI-2010 (optional, only for `code/fi2010.py`)

The external background-structure check uses the NoAuction / decimal-precision files of the FI-2010 benchmark
(Ntakaris et al., 2018), open access under CC BY 4.0:
<https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>

```
data/fi2010/raw/Test_Dst_NoAuction_DecPre_CF_7.txt   (and CF_8, CF_9)
```
