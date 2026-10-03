## Standard Benchmark

| Agent                      | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|----------------------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Advanced (first fact wins) | 2060                | 29615                     | 0.643                  | 0.734              | 1221                    | 0             |
| Advanced (corrections)     | 2049                | 29504                     | 1.000                  | 1.000              | 1215                    | 0             |

## Long-Context Stress Benchmark

| Agent                      | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|----------------------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Advanced (first fact wins) | 391                 | 19080                     | 0.667                  | 0.800              | 986                     | 1             |
| Advanced (corrections)     | 394                 | 19091                     | 1.000                  | 1.000              | 993                     | 1             |
