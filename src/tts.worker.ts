/**
 * TTS Web Worker — Kokoro (inglés con stream progresivo)
 * El español se maneja en el hilo principal via Web Speech API.
 * Auto-detecta WebGPU (5-10x más rápido). Audio empieza en segundos.
 */

import { KokoroTTS, TextSplitterStream } from 'kokoro-js';
import type { RawAudio } from '@huggingface/transformers';

// ── Tipos de mensajes ────────────────────────────────────────────────────────

export type WorkerInMessage =
    | { type: 'generate'; text: string; voice: string }
    | { type: 'cancel' };

export type WorkerOutMessage =
    | { type: 'model_loading'; message: string }
    | { type: 'download_progress'; file: string; progress: number; loaded: number; total: number }
    | { type: 'model_ready'; device: string }
    | { type: 'chunk'; audio: Float32Array; sampleRate: number; index: number }
    | { type: 'done'; totalChunks: number }
    | { type: 'error'; message: string };

// ── Singletons ───────────────────────────────────────────────────────────────

let kokoroInstance: KokoroTTS | null = null;
let kokoroDevice: string = 'wasm';
let cancelled = false;

function send(msg: WorkerOutMessage, transfer?: Transferable[]) {
    self.postMessage(msg, transfer ? { transfer } : undefined);
}

// ── Auto-detección de dispositivo ────────────────────────────────────────────

async function detectDevice(): Promise<{ device: 'webgpu' | 'wasm'; dtype: 'fp32' | 'q8' }> {
    try {
        if ('gpu' in navigator) {
            const adapter = await (navigator as any).gpu.requestAdapter();
            if (adapter) return { device: 'webgpu', dtype: 'fp32' };
        }
    } catch { /* WebGPU no disponible */ }
    return { device: 'wasm', dtype: 'q8' };
}

// ── Inglés: Kokoro con stream() API ─────────────────────────────────────────

async function generateEnglish(text: string, voice: string) {
    if (!kokoroInstance) {
        const { device, dtype } = await detectDevice();
        kokoroDevice = device;
        const deviceLabel = device === 'webgpu' ? '⚡ GPU (WebGPU)' : '🖥 CPU (WebAssembly)';
        send({
            type: 'model_loading',
            message: `Cargando Kokoro TTS con ${deviceLabel}...`,
        });
        kokoroInstance = await KokoroTTS.from_pretrained(
            'onnx-community/Kokoro-82M-v1.0-ONNX',
            {
                dtype,
                device,
                progress_callback: (p: { status: string; name?: string; file?: string; progress?: number; loaded?: number; total?: number }) => {
                    if (p.status === 'progress' && p.file) {
                        send({
                            type: 'download_progress',
                            file: p.file,
                            progress: p.progress ?? 0,
                            loaded: p.loaded ?? 0,
                            total: p.total ?? 0,
                        });
                    }
                },
            }
        );
        send({ type: 'model_ready', device: kokoroDevice });
    }

    const splitter = new TextSplitterStream();
    const stream = kokoroInstance.stream(splitter, {
        voice: voice as Parameters<KokoroTTS['stream']>[1] extends { voice?: infer V } ? V : string,
    });

    ; (async () => {
        splitter.push(text);
        splitter.close();
    })();

    let chunkIndex = 0;
    for await (const { audio } of stream) {
        if (cancelled) return;
        const rawAudio = audio as RawAudio;
        const copy = new Float32Array(rawAudio.audio);
        send(
            { type: 'chunk', audio: copy, sampleRate: rawAudio.sampling_rate, index: chunkIndex++ },
            [copy.buffer]
        );
    }

    if (!cancelled) {
        send({ type: 'done', totalChunks: chunkIndex });
    }
}

// ── Handler ──────────────────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<WorkerInMessage>) => {
    const msg = event.data;

    if (msg.type === 'cancel') {
        cancelled = true;
        return;
    }

    if (msg.type === 'generate') {
        cancelled = false;
        try {
            await generateEnglish(msg.text, msg.voice);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Error inesperado en el worker TTS.';
            kokoroInstance = null;
            send({ type: 'error', message });
        }
    }
};
