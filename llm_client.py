"""
LLM client: Ollama interface with SHA256-keyed disk cache (diskcache).

Usage:
    client = LLMClient("qwen2.5-7b-instruct")
    response = client.generate("Score this candidate…", json_mode=True)
"""

import json
import hashlib
import time
from typing import Optional

import requests
import diskcache

from config import OLLAMA_BASE_URL, CACHE_DIR, LLM_TEMPERATURE, MODELS, EMBEDDING_MODEL


class LLMClient:
    def __init__(self, model_key: str, cache_name: str = "llm_cache"):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS)}")
        self.spec = MODELS[model_key]
        self.model_key = model_key
        self.cache = diskcache.Cache(str(CACHE_DIR / cache_name))
        self._base = OLLAMA_BASE_URL

    # ── Core generation ────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 512,
        skip_cache: bool = False,
    ) -> str:
        cache_key = self._cache_key(prompt, system, json_mode, temperature)
        if not skip_cache and cache_key in self.cache:
            return self.cache[cache_key]

        payload = {
            "model": self.spec.ollama_tag,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = self._post("/api/generate", payload)
        text = resp.get("response", "")
        self.cache[cache_key] = text
        return text

    def chat(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = 512,
        skip_cache: bool = False,
    ) -> str:
        cache_key = self._cache_key(
            json.dumps(messages, ensure_ascii=False), "", json_mode, temperature
        )
        if not skip_cache and cache_key in self.cache:
            return self.cache[cache_key]

        payload = {
            "model": self.spec.ollama_tag,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        resp = self._post("/api/chat", payload)
        text = resp.get("message", {}).get("content", "")
        self.cache[cache_key] = text
        return text

    # ── Embedding extraction (for WEAT/SEAT) ───────────────────────────
    def embed(self, text: str) -> list[float]:
        cache_key = self._cache_key(text, "__embed__", False, 0.0)
        if cache_key in self.cache:
            return self.cache[cache_key]

        payload = {"model": EMBEDDING_MODEL, "input": text}
        resp = self._post("/api/embed", payload)
        emb = resp.get("embeddings", [[]])[0]
        self.cache[cache_key] = emb
        return emb

    # ── Helpers ─────────────────────────────────────────────────────────
    def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        url = f"{self._base}{path}"
        for attempt in range(retries):
            try:
                r = requests.post(url, json=payload, timeout=300)
                if not r.ok:
                    detail = r.text.strip()
                    raise RuntimeError(f"Ollama request failed {r.status_code} for {path} with model={payload.get('model')}: {detail[:500]}")
                return r.json()
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Ollama unreachable after {retries} retries: {e}")
                time.sleep(2 ** attempt)
            except RuntimeError:
                raise
        return {}

    @staticmethod
    def _cache_key(prompt: str, system: str, json_mode: bool, temp: float) -> str:
        raw = f"{prompt}||{system}||{json_mode}||{temp}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear_cache(self):
        self.cache.clear()

    def cache_stats(self) -> dict:
        return {"size": len(self.cache), "volume_mb": self.cache.volume() / 1e6}
