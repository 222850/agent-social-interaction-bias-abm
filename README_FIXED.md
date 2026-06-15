# Fixed launch instructions

Use only `qwen2.5-7b-instruct` in CLI and in the framework.

## One-time setup
```bash
cd ~/bias_framework
export PATH=$HOME/bin:$PATH
nohup ollama serve > ollama.log 2>&1 &
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma
pip install -r requirements.txt
bash setup_data.sh
```

## Verify
```bash
curl http://localhost:11434/api/tags
```
You should see `qwen2.5:7b-instruct` and `embeddinggemma`.

## Run
```bash
python run_evaluation.py --models qwen2.5-7b-instruct --stages jobfair encoded
python run_evaluation.py --models qwen2.5-7b-instruct --stages weat
python run_evaluation.py --models qwen2.5-7b-instruct --stages abm
```

## What was fixed
- model key switched to `qwen2.5-7b-instruct` everywhere
- `/api/embed` now uses `embeddinggemma` instead of the instruct model
- `setup_data.sh` now always writes into the project `data/` directory
- Pingouin p-value extraction made version-robust
- Mesa agents updated for the newer API with `unique_id`
- Ollama error messages now include response text and requested model
