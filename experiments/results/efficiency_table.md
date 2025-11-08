# Efficiency Comparison Table

| Model    |   Sparsity | Pruning Method   |   Top-1 Acc (%) |   Params (M) |   Size (MB) |   Latency (ms) |
|:---------|-----------:|:-----------------|----------------:|-------------:|------------:|---------------:|
| ResNet18 |       0    | N/A (dense)      |            94.9 |        11.17 |       42.7  |            1.9 |
| ResNet18 |       0.5  | Magnitude        |            94.3 |         5.58 |       22.42 |            1.9 |
| ResNet18 |       0.9  | Magnitude        |            92.7 |         1.12 |        6.19 |            1.9 |
| ResNet18 |       0.93 | Magnitude        |            92.5 |         0.84 |        5.18 |            1.9 |
| ResNet18 |       0.5  | Structured       |            94   |         5.58 |       22.42 |            1.3 |
| ResNet18 |       0.9  | Structured       |            90.3 |         1.12 |        6.19 |            0.9 |
| ResNet18 |       0.93 | Structured       |            89.5 |         0.84 |        5.18 |            0.8 |
| ResNet18 |       0.5  | Lottery Ticket   |            94.2 |         5.58 |       22.42 |            1.9 |
| ResNet18 |       0.9  | Lottery Ticket   |            92.5 |         1.12 |        6.19 |            1.9 |
| ResNet18 |       0.93 | Lottery Ticket   |            92.2 |         0.84 |        5.18 |            1.9 |