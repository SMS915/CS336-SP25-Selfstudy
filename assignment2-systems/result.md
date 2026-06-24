## distributed single-node
| backend | device | data size (MB) | nprocs | latency (ms) | bandwidth (GB/s) |
| ------- | ------ | -------------- | ------ | ------------ | ---------------- |
| NCCL    | cuda   | 1              | 2      | 0.9381       | 1.1178           |
| NCCL    | cuda   | 1              | 4      | 0.7635       | 1.3734           |
| NCCL    | cuda   | 1              | 6      | 1.5831       | 0.6624           |
| NCCL    | cuda   | 10             | 2      | 1.4921       | 7.0276           |
| NCCL    | cuda   | 10             | 4      | 7.8454       | 1.3365           |
| NCCL    | cuda   | 10             | 6      | 6.8664       | 1.5271           |
| NCCL    | cuda   | 100            | 2      | 14.4343      | 7.2645           |
| NCCL    | cuda   | 100            | 4      | 55.5107      | 1.8890           |
| NCCL    | cuda   | 100            | 6      | 56.7374      | 1.8481           |
| NCCL    | cuda   | 1000           | 2      | 138.9091     | 7.5487           |
| NCCL    | cuda   | 1000           | 4      | 606.7961     | 1.7281           |
| NCCL    | cuda   | 1000           | 6      | 384.8272     | 2.7248           |
