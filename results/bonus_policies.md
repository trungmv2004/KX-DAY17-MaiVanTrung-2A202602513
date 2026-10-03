## Confidence threshold

| Agent         | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|---------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Threshold 0   | 59                  | 350                       | 0.000                  | 0.200              | 294                     | 0             |
| Threshold 0.8 | 58                  | 348                       | 1.000                  | 1.000              | 290                     | 0             |

## Memory decay

| Agent             | Agent tokens only   | Prompt tokens processed   | Cross-session recall   | Response quality   | Memory growth (bytes)   | Compactions   |
|-------------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Decay off         | 195                 | 2509                      | 1.000                  | 1.000              | 417                     | 0             |
| Half-life 2 turns | 190                 | 2437                      | 0.500                  | 0.600              | 417                     | 0             |
