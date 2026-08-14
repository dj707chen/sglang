# Run a small model locally on MacBook Pro M3

To understand how SGLang runs, how the component processes work together, run the inference of a LLM model locally on this MacBook Pro M3:

1. Choose a small open weight LLM model, serve it via SGLang locally;
   - Tell me how you selected the model;
   - Where you tried to download from.
2. I'm interested in the details of major components (not limited, I may miss some):
   - API server, how the Rust server runs?
   - Tokenizer, TokenizerManager
   - Scheduler
   - TpModelWorker
   - ModelRunner, how many?
3. If you can find one, run a tool to monitor the components / display the status / statistics continuously;
4. Record the network traffic;
5. Set log level so that it logs more details but not overwhelming.
6. Create scripts to be used myself to
  - Manually stop the processes;
  - Re-run the whole serving servers.

---
Add notes below.
