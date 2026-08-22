
## What it looks like under load

    PID COMPONENT                                RSS   CPU  THR      UP
    77398 HTTP server + TokenizerManager          447M    1%   23  17m12s
    77530 └─ resource_tracker (stdlib)             10M    0%    1  17m06s
    77531 └─ Scheduler +TpWorker +ModelRunner   1,609M   58%   57  17m06s
    77532 └─ DetokenizerManager                   344M    0%    7  17m06s

    tokens/s              now  30s   │            p50      p95      p99
    decode              143.8  █▁    │ TTFT     817.4ms   1.41s    2.23s
    prefill (compute)    10.4  █▁    │ inter-t   17.6ms  22.8ms   33.4ms
    requests/s           1.42  ▁█    │ e2e        2.43s   2.94s    2.99s
