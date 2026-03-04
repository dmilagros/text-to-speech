#!/usr/bin/env python3
"""
Coqui TTS Server — XTTS v2 con 28 hablantes × 17 idiomas
Sin API key · Sin límites · Puerto: 3001
"""

import io
import wave
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

PORT = 3001

# ── 28 hablantes integrados en XTTS v2 ───────────────────────────────────────
# Cada uno es una persona real cuya voz fue usada en el entrenamiento.
# Todos son capaces de hablar en cualquiera de los 17 idiomas.

XTTS_SPEAKERS = [
    # Femeninas
    {"id": "Claribel Dervla",    "name": "Claribel",    "gender": "F"},
    {"id": "Daisy Studious",     "name": "Daisy",       "gender": "F"},
    {"id": "Gracie Wise",        "name": "Gracie",      "gender": "F"},
    {"id": "Tammie Ema",         "name": "Tammie",      "gender": "F"},
    {"id": "Alison Dietlinde",   "name": "Alison",      "gender": "F"},
    {"id": "Ana Florence",       "name": "Ana",         "gender": "F"},
    {"id": "Annmarie Nele",      "name": "Annmarie",    "gender": "F"},
    {"id": "Asya Anara",         "name": "Asya",        "gender": "F"},
    {"id": "Brenda Stern",       "name": "Brenda",      "gender": "F"},
    {"id": "Gitta Nikolina",     "name": "Gitta",       "gender": "F"},
    {"id": "Henriette Usha",     "name": "Henriette",   "gender": "F"},
    {"id": "Sofia Hellen",       "name": "Sofia",       "gender": "F"},
    {"id": "Tammy Grit",         "name": "Tammy",       "gender": "F"},
    {"id": "Tanja Adelina",      "name": "Tanja",       "gender": "F"},
    {"id": "Vjollca Johnnie",    "name": "Vjollca",     "gender": "F"},
    # Masculinas
    {"id": "Andrew Chipper",     "name": "Andrew",      "gender": "M"},
    {"id": "Badr Odhiambo",      "name": "Badr",        "gender": "M"},
    {"id": "Dionisio Schuyler",  "name": "Dionisio",    "gender": "M"},
    {"id": "Royston Min",        "name": "Royston",     "gender": "M"},
    {"id": "Viktor Eka",         "name": "Viktor",      "gender": "M"},
    {"id": "Abrahan Mack",       "name": "Abrahan",     "gender": "M"},
    {"id": "Adde Michal",        "name": "Adde",        "gender": "M"},
    {"id": "Baldur Sanjin",      "name": "Baldur",      "gender": "M"},
    {"id": "Craig Gutsy",        "name": "Craig",       "gender": "M"},
    {"id": "Damien Black",       "name": "Damien",      "gender": "M"},
    {"id": "Gilberto Mathias",   "name": "Gilberto",    "gender": "M"},
    {"id": "Ilkin Urbano",       "name": "Ilkin",       "gender": "M"},
    {"id": "Kazuhiko Atallah",   "name": "Kazuhiko",    "gender": "M"},
]

# ── 17 idiomas soportados por XTTS v2 ────────────────────────────────────────

XTTS_LANGUAGES = [
    {"code": "es", "name": "Español",    "flag": "🌎"},
    {"code": "en", "name": "English",    "flag": "🇺🇸"},
    {"code": "fr", "name": "Français",   "flag": "🇫🇷"},
    {"code": "de", "name": "Deutsch",    "flag": "🇩🇪"},
    {"code": "it", "name": "Italiano",   "flag": "🇮🇹"},
    {"code": "pt", "name": "Português",  "flag": "🇧🇷"},
    {"code": "pl", "name": "Polski",     "flag": "🇵🇱"},
    {"code": "tr", "name": "Türkçe",     "flag": "🇹🇷"},
    {"code": "ru", "name": "Русский",    "flag": "🇷🇺"},
    {"code": "nl", "name": "Nederlands", "flag": "🇳🇱"},
    {"code": "cs", "name": "Čeština",    "flag": "🇨🇿"},
    {"code": "ar", "name": "العربية",   "flag": "🇸🇦"},
    {"code": "zh-cn", "name": "中文",   "flag": "🇨🇳"},
    {"code": "hu", "name": "Magyar",     "flag": "🇭🇺"},
    {"code": "ko", "name": "한국어",     "flag": "🇰🇷"},
    {"code": "ja", "name": "日本語",     "flag": "🇯🇵"},
    {"code": "hi", "name": "हिंदी",    "flag": "🇮🇳"},
]

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Coqui XTTS v2 Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_xtts: Any = None


def get_xtts():
    global _xtts
    if _xtts is None:
        from TTS.api import TTS
        print("[XTTS v2] Cargando en CPU...")
        _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print(f"[XTTS v2] ✅ Listo — {len(_xtts.speakers)} hablantes disponibles")
    return _xtts


def samples_to_wav(samples: Any, sample_rate: int) -> bytes:
    arr = np.array(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/speakers")
def get_speakers():
    """Lista de hablantes XTTS v2 con género."""
    return XTTS_SPEAKERS


@app.get("/api/languages")
def get_languages():
    """Lista de idiomas soportados por XTTS v2."""
    return XTTS_LANGUAGES


@app.get("/api/tts")
def tts_endpoint(text: str, speaker: str = "Claribel Dervla", language: str = "es"):
    if not text.strip():
        raise HTTPException(status_code=400, detail="'text' no puede estar vacío.")

    print(f"[TTS] lang={language} speaker={speaker!r} len={len(text)}")

    try:
        tts = get_xtts()
        samples = tts.tts(text=text, speaker=speaker, language=language)
        sample_rate = tts.synthesizer.tts_config.audio["output_sample_rate"]
        wav = samples_to_wav(samples, sample_rate)
        return Response(content=wav, media_type="audio/wav",
                        headers={"Content-Length": str(len(wav))})
    except Exception as e:
        print(f"[TTS] Error: {e}")
        raise HTTPException(status_code=502, detail=f"Error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎙  Coqui XTTS v2 — 28 hablantes × 17 idiomas")
    print(f"   Hablantes: GET http://localhost:{PORT}/api/speakers")
    print(f"   Idiomas:   GET http://localhost:{PORT}/api/languages")
    print(f"   Audio:     GET http://localhost:{PORT}/api/tts?text=Hola&speaker=Claribel+Dervla&language=es")
    print("   (Modelo descarga ~1.8GB la primera vez)\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
