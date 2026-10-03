# Private data layout / 私有数据结构

## English

The source data are not included in this repository because redistribution
permission has not been established. To run the analysis, users must obtain
appropriately licensed minute-level index data and place the following
directories next to the Python scripts:

```text
1-沪深300数据/
2-上证指数数据/
3-深证成指/
```

Expected annual filenames:

```text
1-沪深300数据/1-2015.csv ... 1-2025.csv
2-上证指数数据/2-2015.csv ... 2-2025.csv
3-深证成指/3-2015.csv ... 3-2025.csv
```

Required CSV columns:

```text
INDEX_TICKER
EXCHANGE_CD
TDATE_TIME
OPEN_INDEX
HIGH_INDEX
LOW_INDEX
CLOSE_INDEX
VOLUME
AMOUNT
```

The analysis expects 240 minute observations for each index trading day and
uses timestamps including the Zone 2 and Zone 5 boundaries defined in the code.
Do not commit licensed source files to this repository.

## 中文

由于尚未确认数据再分发许可，本仓库不包含原始数据。运行分析前，使用者必须自行取得具有适当许可的分钟级指数数据，并将以下文件夹放在Python脚本同级目录中：

```text
1-沪深300数据/
2-上证指数数据/
3-深证成指/
```

年度文件命名方式：

```text
1-沪深300数据/1-2015.csv ... 1-2025.csv
2-上证指数数据/2-2015.csv ... 2-2025.csv
3-深证成指/3-2015.csv ... 3-2025.csv
```

CSV必需字段：

```text
INDEX_TICKER
EXCHANGE_CD
TDATE_TIME
OPEN_INDEX
HIGH_INDEX
LOW_INDEX
CLOSE_INDEX
VOLUME
AMOUNT
```

分析程序要求每个指数交易日具有240条分钟记录，并包含代码中定义的二区和五区边界时间。请勿将受许可限制的原始数据提交到本仓库。
