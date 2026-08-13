# Run a small model locally on MacBook Pro M3

To understand how SGLang runs, how the component processes work together, run the inference of a LLM model locally on this MacBook Pro M3:

- Choose a small open weight LLM model, serve it via SGLang locally;
- I'm interested in the details of major components (not limited, I may miss some):
  - API server, how the Rust server runs?
  - Tokenizer, TokenizerManager
  - Scheduler
  - TpModelWorker
  - ModelRunner, how many?
- If you can find one, run a tool to monitor the components / display the status / statistics continuously;
- high log level.
