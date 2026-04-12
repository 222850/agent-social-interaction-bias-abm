# Models

This project uses locally available Ollama models and does not store model weights in the repository.

## Main generation model
- `qwen2.5:7b-instruct`

## Embedding model
- `embeddinggemma`

## Why external models are not stored here
The repository is intended as a lightweight MVP and research prototype. Storing model weights in Git is not practical. Instead, the project expects the user to install the required models locally via Ollama.

## Example setup

ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma