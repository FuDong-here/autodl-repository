# Backtesting

这个目录只放独立回测工具，不修改官方提供的 `code/src` 文件。

`backtest.py` 会复用现有的：

- `StockTransformer`
- 官方特征工程
- 排序数据集构造
- `WeightedRankingLoss`
- 推理阶段的 Top5 逻辑

它的目标是模拟真实提交：站在某个 `signal_date`，只使用当时可见的数据训练，然后预测后面 5 个交易日，最后计算 `visible_proxy`。

## 快速运行

在项目根目录先激活虚拟环境：

```bash
source .venv/bin/activate
```

然后执行：

```bash
python backtesting/backtest.py --folds 1 --epochs 1
```

如果不想激活虚拟环境，也可以直接用项目里的 Python：

```bash
.venv/bin/python backtesting/backtest.py --folds 1 --epochs 1
```

这个命令只跑最近 1 个有效周频窗口，每折训练 1 个 epoch，主要用于检查流程是否能跑通。

## 推荐调试命令

```bash
python backtesting/backtest.py --folds 3 --epochs 3
```

输出文件默认在：

```text
backtesting/results/backtest_results.csv
backtesting/results/backtest_summary.json
```

`backtest_results.csv` 每一行是一轮模拟提交，重点看：

- `portfolio_return`: 当前模型选出的 Top5 等权收益
- `random_return`: 全市场随机平均收益
- `optimal_return`: 事后最优 Top5 平均收益
- `visible_proxy`: 相对随机和事后最优的归一化分数
- `selected_stocks`: 当轮选出的股票

## 常用参数

```bash
python backtesting/backtest.py \
  --data data/train.csv \
  --folds 5 \
  --epochs 5 \
  --batch-size 4 \
  --start 2025-04-27 \
  --end 2026-05-29
```

参数说明：

- `--folds`: 使用最近 N 个有效周频窗口，`0` 表示使用全部窗口。
- `--epochs`: 每个 fold 重新训练的 epoch 数。
- `--start`: 最早评估周开始日期。
- `--end`: 最晚评估周结束日期。
- `--purge`: 训练样本结束日和预测日之间的最小交易日间隔，默认 `5`。
- `--allow-holiday-weeks`: 默认只评估后 5 天是连续自然日的窗口；加上这个参数后允许包含节假日缺口的周。
- `--device`: 可选 `auto`、`cpu`、`cuda`、`mps`。

## 重要说明

当前官方训练标签和数据集构造逻辑是固定未来 5 日收益，所以这个脚本默认并只支持：

```text
horizon = 5
```

如果要回测其他持仓周期，需要先改官方标签构造和窗口构造逻辑。这个版本刻意不做那件事，以保持官方代码不被动到。

完整多折回测会比较慢，因为每个 fold 都会重新做特征工程和训练模型。建议先用 `--folds 1 --epochs 1` 跑通，再逐步增加 folds 和 epochs。
