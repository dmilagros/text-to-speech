/**
 * Piper TTS Worker — Español Latino (es_MX)
 * Usa piper-tts-web con modelos de rhasspy/piper-voices en HuggingFace
 * Respeta puntuación naturalmente, calidad neuronal open source.
 */

import { PiperWebEngine, HuggingFaceVoiceProvider } from 'piper-tts-web';

// ── Tipos de mensajes ─────────────────────────────────────────────────────────

export type PiperWorkerInMessage =
    | { type: 'generate'; text: string; voice: string }
    | { type: 'cancel' };

export type PiperWorkerOutMessage =
    | { type: 'loading'; message: string }
    | { type: 'ready' }
    | { type: 'audio'; wav: ArrayBuffer }
    | { type: 'done' }
    | { type: 'error'; message: string };

// ── Estado ────────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let engine: any | null = null;
let cancelled = false;

function send(msg: PiperWorkerOutMessage, transfer?: Transferable[]) {
    self.postMessage(msg, transfer ? { transfer } : undefined);
}

// ── Handler ───────────────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<PiperWorkerInMessage>) => {
    const msg = event.data;

    if (msg.type === 'cancel') {
        cancelled = true;
        return;
    }

    if (msg.type === 'generate') {
        cancelled = false;
        try {
            if (!engine) {
                send({ type: 'loading', message: 'Cargando motor Piper TTS...' });
                const voiceProvider = new HuggingFaceVoiceProvider();
                engine = new PiperWebEngine({ voiceProvider });
                send({ type: 'ready' });
            }

            send({ type: 'loading', message: `Cargando voz ${msg.voice}... (primera vez ~70MB)` });

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const response: any = await engine.generate(msg.text, msg.voice, 0);

            if (cancelled) return;

            // piper-tts-web devuelve response.audio.buffer (ArrayBuffer WAV)
            const wavBuffer: ArrayBuffer = response?.audio?.buffer ?? response?.buffer ?? response;

            send({ type: 'audio', wav: wavBuffer }, [wavBuffer]);
            send({ type: 'done' });
        } catch (err: unknown) {
            engine = null;
            const message = err instanceof Error ? err.message : 'Error inesperado en Piper TTS.';
            send({ type: 'error', message });
        }
    }
};
