/**
 * TTS Web Worker — Kokoro (inglés, con stream progresivo) + MMS-TTS (español)
 * Auto-detecta WebGPU (5-10x más rápido). Audio empieza en segundos.
 */

import { KokoroTTS, TextSplitterStream } from 'kokoro-js';
import type { RawAudio } from '@huggingface/transformers';
import { pipeline } from '@huggingface/transformers';

// ── Tipos de mensajes ────────────────────────────────────────────────────────

export type WorkerInMessage =
    | { type: 'generate'; text: string; language: 'en' | 'es'; voice: string }
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
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let mmsInstance: any | null = null;
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

    // Usar la API de streaming de Kokoro para audio progresivo
    // Kokoro divide el texto en oraciones internamente — eficiente para cualquier longitud
    const splitter = new TextSplitterStream();
    const stream = kokoroInstance.stream(splitter, {
        voice: voice as Parameters<KokoroTTS['stream']>[1] extends { voice?: infer V } ? V : string,
    });

    // Alimentar texto al splitter en background
    ; (async () => {
        splitter.push(text);
        splitter.close();
    })();

    let chunkIndex = 0;
    for await (const { audio } of stream) {
        if (cancelled) return;
        const rawAudio = audio as RawAudio;
        // Copia el buffer ANTES de transferirlo (transfer lo vacía)
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

// ── Español: MMS-TTS por chunks ──────────────────────────────────────────────

const MMS_CHUNK = 800; // MMS puede manejar fragmentos más grandes

function splitSpanish(input: string): string[] {
    const chunks: string[] = [];
    let current = '';
    const segs = input.split(/([.!?…]+\s*|\n+)/);
    for (const seg of segs) {
        if (!seg) continue;
        if ((current + seg).length > MMS_CHUNK) {
            if (current.trim()) chunks.push(current.trim());
            current = seg;
        } else {
            current += seg;
        }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks.filter(c => c.length > 0);
}

async function generateSpanish(text: string) {
    if (!mmsInstance) {
        send({ type: 'model_loading', message: 'Cargando Meta MMS-TTS español (~100MB)...' });
        mmsInstance = await pipeline('text-to-speech', 'Xenova/mms-tts-spa', {
            device: 'wasm' as never,
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
        } as never);
        send({ type: 'model_ready', device: 'wasm' });
    }

    const chunks = splitSpanish(text);
    let chunkIndex = 0;

    for (let i = 0; i < chunks.length; i++) {
        if (cancelled) return;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result = await mmsInstance(chunks[i], {}) as any;

        let samples: Float32Array | null = null;
        let sampleRate = 16000;

        if (result?.audio instanceof Float32Array) {
            samples = result.audio;
            sampleRate = result.sampling_rate ?? 16000;
        } else if (Array.isArray(result) && result[0]?.audio instanceof Float32Array) {
            samples = result[0].audio;
            sampleRate = result[0].sampling_rate ?? 16000;
        }

        if (!samples) throw new Error('Formato de audio MMS-TTS no reconocido.');

        const copy = new Float32Array(samples);
        send(
            { type: 'chunk', audio: copy, sampleRate, index: chunkIndex++ },
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
            if (msg.language === 'en') await generateEnglish(msg.text, msg.voice);
            else await generateSpanish(msg.text);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Error inesperado en el worker TTS.';
            if (msg.language === 'en') kokoroInstance = null;
            else mmsInstance = null;
            send({ type: 'error', message });
        }
    }
};
