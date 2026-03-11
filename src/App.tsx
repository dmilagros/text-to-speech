/**
 * Voice Studio con Karaoke
 * ─ Local (con npm run server): Coqui XTTS v2 — 28 hablantes × 17 idiomas
 * ─ Deploy (GitHub Pages):     Web Speech API del navegador — voces del sistema
 * ─ Inglés siempre:            Kokoro TTS — calidad premium (streaming)
 * ─ Karaoke: sincronización de palabras durante la reproducción
 */

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import type { WorkerOutMessage, WorkerInMessage } from './tts.worker';
import {
  Volume2, Play, Square, Loader2, Mic2, Settings2,
  AlertCircle, Download, Cpu, Wifi, Server
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

// ── Tipos ─────────────────────────────────────────────────────────────────────

interface XttsSpeaker { id: string; name: string; gender: 'F' | 'M'; }
interface XttsLanguage { code: string; name: string; flag: string; }
interface WSSVoice {
  id: string; name: string; lang: string;
  gender: 'F' | 'M' | '?'; flag: string; langName: string;
  raw: SpeechSynthesisVoice;
}
type TtsMode = 'kokoro' | 'coqui' | 'webspeech';

// ── Karaoke helpers ───────────────────────────────────────────────────────────

/** Divide texto en tokens (palabras + espacios) para karaoke */
function tokenize(text: string): { word: string; isSpace: boolean; charStart: number }[] {
  const tokens: { word: string; isSpace: boolean; charStart: number }[] = [];
  const regex = /(\S+|\s+)/g;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    tokens.push({ word: m[0], isSpace: /^\s+$/.test(m[0]), charStart: m.index });
  }
  return tokens;
}

// ── Web Speech helpers ────────────────────────────────────────────────────────

const LANG_FLAGS: Record<string, string> = {
  es: '🌎', en: '🇺🇸', fr: '🇫🇷', de: '🇩🇪', it: '🇮🇹',
  pt: '🇧🇷', pl: '🇵🇱', tr: '🇹🇷', ru: '🇷🇺', nl: '🇳🇱',
  cs: '🇨🇿', ar: '🇸🇦', zh: '🇨🇳', ja: '🇯🇵', ko: '🇰🇷',
  hi: '🇮🇳', hu: '🇭🇺',
};
const LANG_NAMES: Record<string, string> = {
  es: 'Español', en: 'English', fr: 'Français', de: 'Deutsch', it: 'Italiano',
  pt: 'Português', pl: 'Polski', tr: 'Türkçe', ru: 'Русский', nl: 'Nederlands',
  cs: 'Čeština', ar: 'العربية', zh: 'Chinese', ja: '日本語', ko: '한국어',
  hi: 'हिंदी', hu: 'Magyar',
};

function getWSVoices(): WSSVoice[] {
  const voices = window.speechSynthesis.getVoices();
  const seen = new Set<string>();
  return voices.filter(v => !seen.has(v.voiceURI) && seen.add(v.voiceURI)).map(v => {
    const langCode = v.lang.split('-')[0].toLowerCase();
    const n = v.name.toLowerCase();
    const gender: 'F' | 'M' | '?' =
      /female|mujer|femme|frau|donna|mulher|kvinna|girl|woman/i.test(n) ? 'F' :
        /male|hombre|homme|mann|uomo|homem|man/i.test(n) ? 'M' : '?';
    return {
      id: v.voiceURI, name: v.name, lang: langCode, gender,
      flag: LANG_FLAGS[langCode] ?? '🌐', langName: LANG_NAMES[langCode] ?? langCode.toUpperCase(), raw: v
    };
  });
}

// ── Kokoro voices ─────────────────────────────────────────────────────────────

const KOKORO_VOICES = [
  { id: 'af_heart', label: 'Heart', description: 'Americana · warm ⭐', gender: 'F' },
  { id: 'af_bella', label: 'Bella', description: 'Americana · soft', gender: 'F' },
  { id: 'af_sarah', label: 'Sarah', description: 'Americana · clear', gender: 'F' },
  { id: 'af_nicole', label: 'Nicole', description: 'Americana · ASMR', gender: 'F' },
  { id: 'am_adam', label: 'Adam', description: 'Americano · deep', gender: 'M' },
  { id: 'am_michael', label: 'Michael', description: 'Americano · natural', gender: 'M' },
  { id: 'bf_emma', label: 'Emma', description: 'Británica · elegant', gender: 'F' },
  { id: 'bm_george', label: 'George', description: 'Británico · deep', gender: 'M' },
];

// ── WAV encoder (Kokoro) ──────────────────────────────────────────────────────

/** Concatena chunks de WAV a nivel de bytes (para Coqui streaming). Sin pasar por WebAudio. */
function mergeRawWavParts(parts: Uint8Array[]): Blob {
  if (!parts.length) return new Blob([], { type: 'audio/wav' });
  const firstView = new DataView(parts[0].buffer, parts[0].byteOffset);
  const sampleRate = firstView.getUint32(24, true);
  // Encontrar offset del chunk 'data' en cada parte
  const getPcm = (p: Uint8Array): Uint8Array => {
    const v = new DataView(p.buffer, p.byteOffset);
    let i = 12;
    while (i < p.length - 8) {
      const id = String.fromCharCode(p[i], p[i + 1], p[i + 2], p[i + 3]);
      const sz = v.getUint32(i + 4, true);
      if (id === 'data') return p.slice(i + 8, i + 8 + sz);
      i += 8 + sz;
    }
    return p.slice(44); // fallback: skip standard 44-byte header
  };
  const pcms = parts.map(getPcm);
  const totalPcm = pcms.reduce((s, p) => s + p.byteLength, 0);
  const wavBuf = new ArrayBuffer(44 + totalPcm);
  const v = new DataView(wavBuf); let pos = 0;
  const wS = (s: string) => { for (const c of s) v.setUint8(pos++, c.charCodeAt(0)); };
  const w32 = (d: number) => { v.setUint32(pos, d, true); pos += 4; };
  const w16 = (d: number) => { v.setUint16(pos, d, true); pos += 2; };
  wS('RIFF'); w32(36 + totalPcm); wS('WAVE');
  wS('fmt '); w32(16); w16(1); w16(1); w32(sampleRate); w32(sampleRate * 2); w16(2); w16(16);
  wS('data'); w32(totalPcm);
  const out = new Uint8Array(wavBuf);
  let off = 44;
  for (const p of pcms) { out.set(p, off); off += p.byteLength; }
  return new Blob([wavBuf], { type: 'audio/wav' });
}

function mergeToWavBlob(chunks: { audio: Float32Array; sampleRate: number }[]): Blob {
  if (!chunks.length) return new Blob([], { type: 'audio/wav' });
  const sr = chunks[0].sampleRate;
  const merged = new Float32Array(chunks.reduce((s, c) => s + c.audio.length, 0));
  let off = 0;
  for (const c of chunks) { merged.set(c.audio, off); off += c.audio.length; }
  const ds = merged.length * 2;
  const buf = new ArrayBuffer(44 + ds); const v = new DataView(buf); let p = 0;
  const wS = (s: string) => { for (const c of s) v.setUint8(p++, c.charCodeAt(0)); };
  const w32 = (d: number) => { v.setUint32(p, d, true); p += 4; };
  const w16 = (d: number) => { v.setUint16(p, d, true); p += 2; };
  wS('RIFF'); w32(36 + ds); wS('WAVE');
  wS('fmt '); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
  wS('data'); w32(ds);
  for (let i = 0; i < merged.length; i++) {
    const s = Math.max(-1, Math.min(1, merged[i]));
    v.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true); p += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

// ── KaraokeDisplay ────────────────────────────────────────────────────────────

function KaraokeDisplay({ text, activeChar }: { text: string; activeChar: number }) {
  const tokens = useMemo(() => tokenize(text), [text]);

  // Encuentra el token activo por charIndex
  const activeIdx = useMemo(() => {
    if (activeChar < 0) return -1;
    let best = -1;
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (!t.isSpace && t.charStart <= activeChar) best = i;
    }
    return best;
  }, [activeChar, tokens]);

  // Palabras sin espacios — flex wrap para que el fontSize las reposicione correctamente
  const wordTokens = useMemo(() => tokens.filter(t => !t.isSpace), [tokens]);

  // Recalcula activeIdx indexado sobre wordTokens
  const activeWordIdx = useMemo(() => {
    if (activeChar < 0) return -1;
    let best = -1;
    for (let i = 0; i < wordTokens.length; i++) {
      if (wordTokens[i].charStart <= activeChar) best = i;
    }
    return best;
  }, [activeChar, wordTokens]);

  return (
    <div className="flex flex-wrap justify-center items-start content-start gap-x-3 gap-y-2 px-2 py-4 select-none h-[200px] overflow-y-auto">
      {wordTokens.map((t, i) => {
        const isActive = i === activeWordIdx;
        const isPast = !isActive && activeWordIdx >= 0 && i < activeWordIdx;
        return (
          <motion.span
            key={i}
            animate={{
              opacity: isActive ? 1 : isPast ? 0.22 : 0.55,
              color: isActive ? '#5A5A40' : '#1a1a1a',
              textShadow: isActive
                ? '0 0 20px rgba(90,90,64,0.45), 0 0 8px rgba(90,90,64,0.3)'
                : '0 0 0px transparent',
            }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            style={{ display: 'inline-block', fontWeight: 600 }}
          >
            {t.word}
          </motion.span>
        );
      })}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [text, setText] = useState(
    'Hola, bienvenida a Voice Studio. Esta aplicación convierte texto en voz natural con múltiples voces e idiomas.'
  );

  const [mode, setMode] = useState<TtsMode>('coqui');

  // Coqui state
  const [coquiSpeakers, setCoquiSpeakers] = useState<XttsSpeaker[]>([]);
  const [coquiLanguages, setCoquiLanguages] = useState<XttsLanguage[]>([]);
  const [coquiLang, setCoquiLang] = useState('es');
  const [coquiSpeaker, setCoquiSpeaker] = useState('Claribel Dervla');
  const [coquiGender, setCoquiGender] = useState<'all' | 'F' | 'M'>('all');
  const [serverOk, setServerOk] = useState<boolean | null>(null);

  // Web Speech API state
  const [wsvVoices, setWsvVoices] = useState<WSSVoice[]>([]);
  const [wsvLang, setWsvLang] = useState('es');
  const [wsvVoice, setWsvVoice] = useState('');
  const [wsvGender, setWsvGender] = useState<'all' | 'F' | 'M'>('all');

  // Kokoro state
  const [kokoroVoice, setKokoroVoice] = useState('af_heart');

  // General
  const [status, setStatus] = useState<'idle' | 'loading' | 'generating' | 'playing'>('idle');
  const [statusMsg, setStatusMsg] = useState('');
  const [chunksReceived, setChunksReceived] = useState(0);
  const [deviceInfo, setDeviceInfo] = useState('');
  const [downloadPct, setDownloadPct] = useState(0);
  const [downloadMB, setDownloadMB] = useState('');
  const [estimatedTotal, setEstimatedTotal] = useState(0);
  const [eta, setEta] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Karaoke state
  const [activeChar, setActiveChar] = useState(-1);
  const [isKaraoke, setIsKaraoke] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextStartRef = useRef<number>(0);
  const receivedChunksRef = useRef<{ audio: Float32Array; sampleRate: number }[]>([]);
  const downloadRef = useRef<string | null>(null);
  const genStartRef = useRef<number>(0);
  const abortRef = useRef<AbortController | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const karaokeAudioRef = useRef<HTMLAudioElement | null>(null);
  // Ref estable de startTimeKaraoke para no recrear el worker al cambiar el texto
  const startTimeKaraokeRef = useRef<(el: HTMLAudioElement) => void>(() => { });

  // Tokens memoizados para karaoke por tiempo (palabras no-espacio)
  const wordsOnly = useMemo(() => tokenize(text).filter(t => !t.isSpace), [text]);

  // ── Karaoke por tiempo (para Coqui y Kokoro) ──────────────────────────────

  const startTimeKaraoke = useCallback((audioEl: HTMLAudioElement) => {
    karaokeAudioRef.current = audioEl;
    setIsKaraoke(true);

    const tick = () => {
      const el = karaokeAudioRef.current;
      if (!el || el.paused || el.ended) return;
      const frac = el.currentTime / (el.duration || 1);
      const wordIdx = Math.min(Math.floor(frac * wordsOnly.length), wordsOnly.length - 1);
      setActiveChar(wordsOnly[wordIdx]?.charStart ?? 0);
      rafRef.current = requestAnimationFrame(tick);
    };

    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);

    audioEl.onended = () => {
      cancelAnimationFrame(rafRef.current);
      setActiveChar(-1);
      setIsKaraoke(false);
      setStatus('idle');
      setStatusMsg('');
    };
  }, [wordsOnly]);

  const stopKaraoke = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    setActiveChar(-1);
    setIsKaraoke(false);
  }, []);

  // Mantener ref sincronizada para el worker (evita recreación)
  useEffect(() => { startTimeKaraokeRef.current = startTimeKaraoke; }, [startTimeKaraoke]);

  // ── Cargar speakers + languages ───────────────────────────────────────────

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/speakers`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
      fetch(`${API_BASE}/api/languages`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
    ]).then(([spks, langs]: [XttsSpeaker[], XttsLanguage[]]) => {
      setCoquiSpeakers(spks);
      setCoquiLanguages(langs);
      setServerOk(true);
    }).catch(() => {
      setServerOk(false);
      setMode('webspeech');
    });
  }, []);

  useEffect(() => {
    const load = () => {
      const voices = getWSVoices();
      if (voices.length > 0) {
        setWsvVoices(voices);
        const firstEs = voices.find(v => v.lang === 'es');
        if (firstEs) setWsvVoice(firstEs.id);
      }
    };
    load();
    window.speechSynthesis.onvoiceschanged = load;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────────

  const coquiCurrentLang = coquiLanguages.find(l => l.code === coquiLang);
  const coquiCurrentSpeaker = coquiSpeakers.find(s => s.id === coquiSpeaker);
  const coquiFemale = coquiSpeakers.filter(s => s.gender === 'F');
  const coquiMale = coquiSpeakers.filter(s => s.gender === 'M');

  const wsvLangs = [...new Set(wsvVoices.map(v => v.lang))].sort();
  const wsvFiltered = wsvVoices.filter(v => v.lang === wsvLang && (wsvGender === 'all' || v.gender === wsvGender || v.gender === '?'));

  // ── AudioContext (Kokoro) ─────────────────────────────────────────────────

  const getAudioCtx = useCallback(() => {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContext(); nextStartRef.current = 0;
    }
    if (audioCtxRef.current.state === 'suspended') audioCtxRef.current.resume();
    return audioCtxRef.current;
  }, []);

  const playChunk = useCallback((samples: Float32Array, sampleRate: number) => {
    const ctx = getAudioCtx();
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer; src.connect(ctx.destination);
    const start = Math.max(nextStartRef.current, ctx.currentTime + 0.02);
    src.start(start); nextStartRef.current = start + buffer.duration;
  }, [getAudioCtx]);

  // ── Worker (Kokoro) ───────────────────────────────────────────────────────

  useEffect(() => {
    const worker = new Worker(new URL('./tts.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;
      switch (msg.type) {
        case 'model_loading': setStatus('loading'); setStatusMsg(msg.message); setDownloadPct(0); break;
        case 'download_progress': {
          setDownloadPct(Math.round(msg.progress));
          setDownloadMB(`${(msg.loaded / 1024 / 1024).toFixed(1)} / ${msg.total > 0 ? (msg.total / 1024 / 1024).toFixed(0) : '?'} MB`);
          break;
        }
        case 'model_ready':
          setDeviceInfo(msg.device === 'webgpu' ? '⚡ GPU' : '🖥 CPU');
          setStatus('generating'); setStatusMsg(''); setDownloadPct(0);
          genStartRef.current = Date.now(); break;
        case 'chunk': {
          playChunk(msg.audio, msg.sampleRate);
          receivedChunksRef.current.push({ audio: msg.audio, sampleRate: msg.sampleRate });
          const n = receivedChunksRef.current.length;
          setChunksReceived(n);
          const elapsed = (Date.now() - genStartRef.current) / 1000;
          const rate = elapsed > 0 ? n / elapsed : 0;
          if (rate > 0) setEta(prev => {
            const left = Math.max(0, (estimatedTotal > 0 ? estimatedTotal : n) - n) / rate;
            return left < 5 ? 'casi listo...' : left < 60 ? `~${Math.round(left)}s` : `~${Math.floor(left / 60)}m`;
            return prev;
          });
          if (msg.index === 0) setStatus('playing');
          break;
        }
        case 'done': {
          const blob = mergeToWavBlob(receivedChunksRef.current);
          const url = URL.createObjectURL(blob);
          if (downloadRef.current) URL.revokeObjectURL(downloadRef.current);
          downloadRef.current = url;
          setAudioUrl(url);
          // El WebAudio (streaming) ya reprodujo el audio.
          // Solo guardamos la URL para el player — karaoke se activa si el usuario da play allí.
          setStatus('idle');
          setStatusMsg('');
          break;
        }
        case 'error': setStatus('idle'); setStatusMsg(''); setError(msg.message); break;
      }
    };
    workerRef.current = worker;
    return () => { worker.terminate(); };
  }, [playChunk]);

  // ── Generar audio ─────────────────────────────────────────────────────────

  const generateCoqui = useCallback(async () => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setStatus('loading'); setError(null); setAudioUrl(null); stopKaraoke();
    setStatusMsg(`${coquiCurrentSpeaker?.name ?? coquiSpeaker} · ${coquiCurrentLang?.name ?? coquiLang}`);

    const allChunks: { audio: Float32Array; sampleRate: number }[] = [];
    const rawWavParts: Uint8Array[] = []; // para descarga robusta

    try {
      const res = await fetch(`${API_BASE}/api/tts-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speaker: coquiSpeaker, language: coquiLang }),
        signal: abortRef.current.signal,
      });
      if (!res.ok) { const e = await res.json().catch(() => ({ detail: `HTTP ${res.status}` })); throw new Error(e.detail); }

      const reader = res.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';
      let ctx = audioCtxRef.current;
      if (!ctx || ctx.state === 'closed') { ctx = new AudioContext(); audioCtxRef.current = ctx; }
      let nextStart = ctx.currentTime + 0.1;

      // Schedule de karaoke: [{ startTime (AudioContext), charStart }]
      const schedule: { startTime: number; charStart: number }[] = [];
      let rafId = 0;
      const tickKaraoke = () => {
        const now = audioCtxRef.current?.currentTime ?? 0;
        let active = -1;
        for (const s of schedule) {
          if (s.startTime <= now) active = s.charStart;
          else break;
        }
        if (active >= 0) setActiveChar(active);
        rafId = requestAnimationFrame(tickKaraoke);
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line) as {
            done?: boolean; index?: number; total?: number;
            charStart?: number; text?: string; audio?: string; error?: string;
          };
          if (msg.done) break;
          if (msg.error) { console.warn('[Coqui stream]', msg.error); continue; }
          if (!msg.audio) continue;

          // Decodificar base64 → ArrayBuffer → AudioBuffer
          const raw = atob(msg.audio);
          const ab = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i++) ab[i] = raw.charCodeAt(i);
          // Guardar bytes RAW para descarga (antes de decodeAudioData)
          rawWavParts.push(new Uint8Array(ab));

          // Guardar Float32 para Kokoro-compat
          const audioBuffer = await ctx.decodeAudioData(ab.buffer.slice(0));
          const pcm = new Float32Array(audioBuffer.length);
          audioBuffer.copyFromChannel(pcm, 0);
          allChunks.push({ audio: pcm, sampleRate: audioBuffer.sampleRate });

          // Programar reproducción sin gap
          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(ctx.destination);
          const start = Math.max(nextStart, ctx.currentTime + 0.05);
          const chunkCharStart = msg.charStart ?? 0;
          const chunkText = msg.text ?? '';
          const chunkCharEnd = chunkCharStart + chunkText.length;
          const dur = audioBuffer.duration;

          // Distribuir cada PALABRA de esta oración en el schedule
          // para sincronización word-level dentro de la duración del chunk
          const sentWords = wordsOnly.filter(
            w => w.charStart >= chunkCharStart && w.charStart < chunkCharEnd
          );
          if (sentWords.length > 0) {
            const wordDur = dur / sentWords.length;
            sentWords.forEach((w, wi) => {
              schedule.push({ startTime: start + wi * wordDur, charStart: w.charStart });
            });
          } else {
            // Sin palabras reconocidas → marcar al inicio del chunk
            schedule.push({ startTime: start, charStart: chunkCharStart });
          }

          src.start(start);
          nextStart = start + dur;

          if (allChunks.length === 1) {
            setStatus('playing');
            setIsKaraoke(true);
            setStatusMsg(`${coquiCurrentSpeaker?.name ?? coquiSpeaker} · ${coquiCurrentLang?.name ?? coquiLang}`);
            // Arrancar el rAF loop de karaoke
            rafId = requestAnimationFrame(tickKaraoke);
          }
          setChunksReceived(allChunks.length);
          // ETA
          if (msg.total && msg.total > 1) {
            const done = (msg.index ?? 0) + 1;
            const remaining = msg.total - done;
            setEta(remaining > 0 ? `~${remaining} oración${remaining > 1 ? 'es' : ''}` : '');
          }
        }
      }

      // Esperar que termine el último chunk y limpiar
      const totalDuration = nextStart - (audioCtxRef.current?.currentTime ?? 0);
      setTimeout(() => {
        cancelAnimationFrame(rafId);
        setStatus('idle'); setStatusMsg(''); setEta('');
        setActiveChar(-1); setIsKaraoke(false);
      }, Math.max(0, totalDuration * 1000 + 300));

      // WAV completo para descarga (bytes RAW — fiable en HF Spaces)
      if (rawWavParts.length > 0) {
        const blob = mergeRawWavParts(rawWavParts);
        const url = URL.createObjectURL(blob);
        if (downloadRef.current) URL.revokeObjectURL(downloadRef.current);
        downloadRef.current = url; setAudioUrl(url);
      }
    } catch (err: unknown) {
      if ((err as Error)?.name === 'AbortError') return;
      setStatus('idle'); setStatusMsg(''); stopKaraoke();
      setError(`${err instanceof Error ? err.message : String(err)}`);
    }
  }, [text, coquiSpeaker, coquiLang, coquiCurrentSpeaker, coquiCurrentLang, stopKaraoke, wordsOnly]);


  const generateWebSpeech = useCallback(() => {
    if (!text.trim()) return;
    window.speechSynthesis.cancel();
    const voice = wsvVoices.find(v => v.id === wsvVoice);
    if (!voice) { setError('Selecciona una voz'); return; }
    const utt = new SpeechSynthesisUtterance(text);
    utt.voice = voice.raw; utt.lang = voice.raw.lang;
    setStatus('playing'); setError(null); setAudioUrl(null);
    setIsKaraoke(true);

    // Web Speech API: sincronización EXACTA por evento onboundary
    utt.onboundary = (e: SpeechSynthesisEvent) => {
      if (e.name === 'word') setActiveChar(e.charIndex);
    };
    utt.onend = () => {
      setStatus('idle');
      setActiveChar(-1);
      setIsKaraoke(false);
    };
    utt.onerror = () => { setStatus('idle'); setError('Error Web Speech API'); stopKaraoke(); };
    window.speechSynthesis.speak(utt);
  }, [text, wsvVoice, wsvVoices, stopKaraoke]);

  const generateKokoro = useCallback(() => {
    if (!text.trim() || !workerRef.current) return;
    setEta(''); setStatus('loading'); setError(null); setStatusMsg('Iniciando...');
    setChunksReceived(0); receivedChunksRef.current = []; setAudioUrl(null); setDownloadPct(0);
    setEstimatedTotal(Math.max(1, Math.ceil(text.length / 120)));
    stopKaraoke();
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    nextStartRef.current = 0;
    workerRef.current.postMessage({ type: 'generate', text, voice: kokoroVoice } satisfies WorkerInMessage);
  }, [text, kokoroVoice, stopKaraoke]);

  const generate = useCallback(() => {
    if (mode === 'coqui') generateCoqui();
    else if (mode === 'webspeech') generateWebSpeech();
    else generateKokoro();
  }, [mode, generateCoqui, generateWebSpeech, generateKokoro]);

  const cancel = useCallback(() => {
    if (mode === 'coqui') { abortRef.current?.abort(); audioElRef.current?.pause(); }
    else if (mode === 'webspeech') window.speechSynthesis.cancel();
    else { workerRef.current?.postMessage({ type: 'cancel' } satisfies WorkerInMessage); audioCtxRef.current?.close(); audioCtxRef.current = null; }
    stopKaraoke();
    setStatus('idle'); setStatusMsg(''); setChunksReceived(0);
  }, [mode, stopKaraoke]);

  const handleDownload = () => {
    const url = downloadRef.current || audioUrl;
    if (!url) return;
    const label = mode === 'kokoro' ? kokoroVoice : mode === 'coqui' ? `${coquiSpeaker}_${coquiLang}` : wsvVoice;
    const a = document.createElement('a');
    a.href = url; a.download = `voice-studio-${label}-${Date.now()}.wav`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };

  const isActive = status !== 'idle';

  const engineLabel =
    mode === 'kokoro' ? `Kokoro TTS${deviceInfo ? ` · ${deviceInfo}` : ''}` :
      mode === 'coqui' ? 'Coqui XTTS v2 · CPU' :
        `Web Speech API${wsvVoices.length ? ` · ${wsvVoices.length} voces` : ''}`;

  return (
    <div className="min-h-screen bg-[#f5f2ed] text-[#1a1a1a] font-sans selection:bg-[#5A5A40]/20">
      <div className="max-w-5xl mx-auto px-5 py-12">

        {/* Header */}
        <header className="mb-10 text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center justify-center p-3 bg-white rounded-2xl shadow-sm border border-black/5 mb-5">
            <Volume2 className="w-7 h-7 text-[#5A5A40]" />
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="text-4xl md:text-5xl font-serif font-light mb-3 tracking-tight">
            Voice Studio
          </motion.h1>
          <p className="text-base text-[#1a1a1a]/50 mb-3">Texto a voz · Karaoke · Open source</p>
          <div className="inline-flex items-center gap-2 text-xs text-[#5A5A40]/70 bg-[#5A5A40]/8 px-3 py-1.5 rounded-full">
            <Cpu className="w-3 h-3" />{engineLabel}
          </div>
          {serverOk !== null && mode !== 'kokoro' && (
            <div className={`mt-2 inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full ${serverOk ? 'text-green-700 bg-green-50' : 'text-amber-700 bg-amber-50'}`}>
              {serverOk ? <Server className="w-3 h-3" /> : <Wifi className="w-3 h-3" />}
              {serverOk ? `Coqui XTTS v2 · ${coquiSpeakers.length} hablantes · ${coquiLanguages.length} idiomas` : `Web Speech API (para Coqui: npm run server)`}
            </div>
          )}
        </header>

        <div className="grid gap-6">

          {/* ── VISTA KARAOKE — siempre visible ─────────────────────────────── */}
          <motion.section
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-3xl p-8 shadow-sm border border-black/5 relative overflow-hidden">

            {/* Pulso de fondo animado (solo durante reproducción) */}
            {isKaraoke && (
              <motion.div
                className="absolute inset-0 pointer-events-none"
                animate={{ background: ['rgba(90,90,64,0.03)', 'rgba(90,90,64,0.08)', 'rgba(90,90,64,0.03)'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              />
            )}

            <div className="flex items-center gap-2 mb-4 text-[#5A5A40]">
              {isKaraoke ? (
                <motion.div animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 0.8, repeat: Infinity }}
                  className="w-2 h-2 rounded-full bg-[#5A5A40]" />
              ) : (
                <div className="w-2 h-2 rounded-full bg-[#5A5A40]/30" />
              )}
              <span className="text-xs font-semibold uppercase tracking-widest">
                {isKaraoke ? 'Reproduciendo' : 'Karaoke'}
              </span>
            </div>

            <div className="text-xl md:text-2xl min-h-[6rem] flex items-center justify-center">
              <KaraokeDisplay text={text} activeChar={activeChar} />
            </div>
          </motion.section>

          {/* Engine selector */}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
            className="bg-white rounded-3xl p-2 shadow-sm border border-black/5 flex gap-2 flex-wrap">
            {([
              { id: 'coqui' as TtsMode, label: '🤖 Coqui XTTS v2', sub: '28 voces × 17 idiomas · Local', disabled: serverOk === false },
              { id: 'webspeech' as TtsMode, label: '🌐 Web Speech API', sub: 'Voces del navegador · Siempre', disabled: false },
              { id: 'kokoro' as TtsMode, label: '⭐ Kokoro TTS', sub: 'Inglés premium · Streaming', disabled: false },
            ] as { id: TtsMode; label: string; sub: string; disabled: boolean }[]).map(opt => (
              <button key={opt.id} onClick={() => !opt.disabled && setMode(opt.id)} disabled={opt.disabled}
                className={`flex-1 py-3 px-4 rounded-2xl text-sm transition-all ${mode === opt.id ? 'bg-[#5A5A40] text-white shadow' : opt.disabled ? 'text-[#1a1a1a]/20 cursor-not-allowed' : 'text-[#1a1a1a]/60 hover:text-[#1a1a1a] hover:bg-[#f9f8f6]'}`}>
                <div className="font-medium">{opt.label}</div>
                <div className={`text-xs mt-0.5 ${mode === opt.id ? 'text-white/70' : 'text-[#1a1a1a]/40'}`}>{opt.sub}</div>
              </button>
            ))}
          </motion.div>

          <div className="grid md:grid-cols-5 gap-6">

            {/* ── Panel de voces ─────────────────────────────────────────── */}
            <motion.section initial={{ opacity: 0, x: -15 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
              className="md:col-span-3 bg-white rounded-3xl p-6 shadow-sm border border-black/5">
              <div className="flex items-center gap-2 mb-5 text-[#5A5A40]">
                <Settings2 className="w-4 h-4" />
                <h2 className="text-sm font-semibold uppercase tracking-widest">Voz</h2>
              </div>

              <AnimatePresence mode="wait">

                {/* Kokoro */}
                {mode === 'kokoro' && (
                  <motion.div key="kokoro" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {(['F', 'M'] as const).map(g => (
                      <div key={g} className="mb-4">
                        <p className="text-xs text-[#1a1a1a]/40 uppercase tracking-wider font-medium mb-2">{g === 'F' ? '♀ Femeninas' : '♂ Masculinas'}</p>
                        <div className="grid grid-cols-2 gap-2">
                          {KOKORO_VOICES.filter(v => v.gender === g).map(v => (
                            <button key={v.id} onClick={() => setKokoroVoice(v.id)}
                              className={`flex flex-col p-3 rounded-2xl border text-left transition-all ${kokoroVoice === v.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white' : 'bg-[#f9f8f6] border-black/5 hover:border-[#5A5A40]/30'}`}>
                              <span className="font-medium text-sm">{v.label}</span>
                              <span className={`text-xs mt-0.5 ${kokoroVoice === v.id ? 'text-white/70' : 'text-[#1a1a1a]/50'}`}>{v.description}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </motion.div>
                )}

                {/* Coqui */}
                {mode === 'coqui' && (
                  <motion.div key="coqui" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <p className="text-xs text-[#1a1a1a]/40 uppercase tracking-wider font-medium mb-2">Idioma</p>
                    <div className="flex flex-wrap gap-1.5 mb-4">
                      {coquiLanguages.map(lang => (
                        <button key={lang.code} onClick={() => setCoquiLang(lang.code)}
                          className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${coquiLang === lang.code ? 'bg-[#5A5A40] border-[#5A5A40] text-white' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a]/70 hover:border-[#5A5A40]/30'}`}>
                          <span>{lang.flag}</span><span>{lang.name}</span>
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs text-[#1a1a1a]/40 uppercase tracking-wider font-medium">Hablante</p>
                      <div className="flex gap-1">
                        {(['all', 'F', 'M'] as const).map(g => (
                          <button key={g} onClick={() => setCoquiGender(g)}
                            className={`px-2.5 py-1 rounded-full text-xs font-medium transition-all ${coquiGender === g ? 'bg-[#5A5A40] text-white' : 'bg-[#f9f8f6] text-[#1a1a1a]/50 hover:bg-[#eee]'}`}>
                            {g === 'all' ? 'Todos' : g === 'F' ? '♀' : '♂'}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 max-h-56 overflow-y-auto pr-1">
                      {(coquiGender !== 'M' ? coquiFemale : []).map(sp => (
                        <button key={sp.id} onClick={() => setCoquiSpeaker(sp.id)}
                          className={`flex flex-col items-start p-2.5 rounded-xl border text-left transition-all ${coquiSpeaker === sp.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow' : 'bg-[#f9f8f6] border-black/5 hover:border-[#5A5A40]/30'}`}>
                          <span className={`text-[10px] mb-0.5 ${coquiSpeaker === sp.id ? 'text-white/60' : 'text-pink-400'}`}>♀</span>
                          <span className="font-medium text-xs">{sp.name}</span>
                        </button>
                      ))}
                      {(coquiGender !== 'F' ? coquiMale : []).map(sp => (
                        <button key={sp.id} onClick={() => setCoquiSpeaker(sp.id)}
                          className={`flex flex-col items-start p-2.5 rounded-xl border text-left transition-all ${coquiSpeaker === sp.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow' : 'bg-[#f9f8f6] border-black/5 hover:border-[#5A5A40]/30'}`}>
                          <span className={`text-[10px] mb-0.5 ${coquiSpeaker === sp.id ? 'text-white/60' : 'text-blue-400'}`}>♂</span>
                          <span className="font-medium text-xs">{sp.name}</span>
                        </button>
                      ))}
                    </div>
                    {coquiSpeaker && (
                      <p className="text-xs text-[#1a1a1a]/40 mt-3">
                        <span className="font-medium text-[#5A5A40]">{coquiCurrentSpeaker?.gender === 'F' ? '♀' : '♂'} {coquiSpeaker}</span>
                        {' · '}{coquiCurrentLang?.flag} {coquiCurrentLang?.name}
                      </p>
                    )}
                  </motion.div>
                )}

                {/* Web Speech API */}
                {mode === 'webspeech' && (
                  <motion.div key="wsv" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {wsvVoices.length === 0 ? (
                      <p className="text-sm text-[#1a1a1a]/40 text-center py-8">Cargando voces del sistema...</p>
                    ) : (
                      <>
                        <p className="text-xs text-[#1a1a1a]/40 uppercase tracking-wider font-medium mb-2">Idioma ({wsvLangs.length})</p>
                        <div className="flex flex-wrap gap-1.5 mb-4">
                          {wsvLangs.map(lang => {
                            const vv = wsvVoices.find(x => x.lang === lang);
                            return (
                              <button key={lang} onClick={() => { setWsvLang(lang); setWsvVoice(wsvVoices.find(x => x.lang === lang)?.id ?? ''); }}
                                className={`flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${wsvLang === lang ? 'bg-[#5A5A40] border-[#5A5A40] text-white' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a]/70 hover:border-[#5A5A40]/30'}`}>
                                <span>{vv?.flag ?? '🌐'}</span><span>{vv?.langName ?? (lang as string).toUpperCase()}</span>
                              </button>
                            );
                          })}
                        </div>
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs text-[#1a1a1a]/40 uppercase tracking-wider font-medium">Voz ({wsvFiltered.length})</p>
                          <div className="flex gap-1">
                            {(['all', 'F', 'M'] as const).map(g => (
                              <button key={g} onClick={() => setWsvGender(g)}
                                className={`px-2.5 py-1 rounded-full text-xs font-medium transition-all ${wsvGender === g ? 'bg-[#5A5A40] text-white' : 'bg-[#f9f8f6] text-[#1a1a1a]/50 hover:bg-[#eee]'}`}>
                                {g === 'all' ? 'Todas' : g === 'F' ? '♀' : '♂'}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                          {wsvFiltered.map(v => (
                            <button key={v.id} onClick={() => setWsvVoice(v.id)}
                              className={`w-full flex items-center gap-3 p-3 rounded-xl border text-left transition-all ${wsvVoice === v.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white' : 'bg-[#f9f8f6] border-black/5 hover:border-[#5A5A40]/30'}`}>
                              <span className={`text-sm ${wsvVoice === v.id ? 'text-white/70' : v.gender === 'F' ? 'text-pink-400' : v.gender === 'M' ? 'text-blue-400' : 'text-[#1a1a1a]/30'}`}>
                                {v.gender === 'F' ? '♀' : v.gender === 'M' ? '♂' : '?'}
                              </span>
                              <span className="font-medium text-sm truncate">{v.name}</span>
                            </button>
                          ))}
                          {wsvFiltered.length === 0 && (
                            <p className="text-xs text-[#1a1a1a]/40 text-center py-4">No hay voces para este idioma en tu sistema</p>
                          )}
                        </div>
                        <p className="text-xs text-[#1a1a1a]/35 mt-3 italic">Las voces dependen del sistema operativo.</p>
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>

            {/* ── Texto + Botón ────────────────────────────────────────────── */}
            <div className="md:col-span-2 flex flex-col gap-5">

              <motion.section initial={{ opacity: 0, x: 15 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.35 }}
                className="bg-white rounded-3xl p-6 shadow-sm border border-black/5">
                <div className="flex items-center gap-2 mb-4 text-[#5A5A40]">
                  <Mic2 className="w-4 h-4" />
                  <h2 className="text-sm font-semibold uppercase tracking-widest">Texto</h2>
                </div>
                <textarea value={text} onChange={e => setText(e.target.value)}
                  placeholder="Escribe aquí para convertir a voz..."
                  className="w-full h-36 p-3 bg-[#f9f8f6] rounded-2xl border border-black/5 focus:outline-none focus:ring-2 focus:ring-[#5A5A40]/20 resize-none text-sm leading-relaxed" />
                <div className="mt-1.5 text-right text-xs text-[#1a1a1a]/35">{text.length.toLocaleString()} chars</div>
              </motion.section>

              <motion.button initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}
                onClick={isActive ? cancel : generate}
                disabled={!text.trim() && !isActive}
                className="bg-[#5A5A40] hover:bg-[#4a4a34] text-white rounded-3xl p-7 flex flex-col items-center justify-center gap-3 transition-all shadow-lg active:scale-95 relative overflow-hidden disabled:opacity-50">
                {status === 'loading' && downloadPct > 0 && (
                  <motion.div className="absolute bottom-0 left-0 h-1.5 bg-white/40"
                    animate={{ width: `${downloadPct}%` }} transition={{ ease: 'easeOut', duration: 0.3 }} />
                )}
                {(status === 'generating' || status === 'playing') && (
                  <motion.div className="absolute bottom-0 left-0 h-1 bg-white/25"
                    animate={{ width: ['0%', '100%'] }} transition={{ duration: 3, repeat: Infinity, ease: 'linear' }} />
                )}
                {isActive ? <Loader2 className="w-10 h-10 animate-spin" /> : <Play className="w-10 h-10 fill-current" />}
                <div className="text-center">
                  <span className="text-base font-serif italic block mb-1">
                    {status === 'idle' && 'Generar Voz'}
                    {status === 'loading' && (statusMsg || 'Cargando...')}
                    {status === 'generating' && 'Generando...'}
                    {status === 'playing' && '▶ Karaoke activo...'}
                  </span>
                  {mode === 'kokoro' && (status === 'generating' || status === 'playing') && chunksReceived > 0 && (
                    <p className="text-xs text-white/60">{chunksReceived}/{estimatedTotal} frases{eta ? ` · ${eta}` : ''}</p>
                  )}
                  {status === 'loading' && downloadPct > 0 && (
                    <div className="w-full px-2">
                      <div className="flex justify-between text-xs text-white/60 mb-1">
                        <span className="font-bold">{downloadPct}%</span><span>{downloadMB}</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-1.5">
                        <motion.div className="bg-white/60 h-1.5 rounded-full" animate={{ width: `${downloadPct}%` }} />
                      </div>
                    </div>
                  )}
                  {isActive && <p className="text-xs text-white/30 mt-1">Toca para cancelar</p>}
                </div>
                {isActive && <Square className="w-3 h-3 absolute top-4 right-4 opacity-30" />}
              </motion.button>
            </div>
          </div>

          {/* Audio output */}
          <AnimatePresence>
            {(audioUrl || error) && (
              <motion.section initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className={`rounded-3xl p-6 border ${error ? 'bg-red-50 border-red-100' : 'bg-white border-black/5 shadow-sm'}`}>
                {error ? (
                  <div className="flex items-start gap-3 text-red-600">
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-sm">{error}</p>
                      {mode === 'coqui' && (
                        <p className="text-xs mt-1 text-red-400">
                          ¿Está corriendo el servidor? <code className="bg-red-100 px-1 rounded">npm run server</code>
                        </p>
                      )}
                    </div>
                  </div>
                ) : audioUrl ? (
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-serif italic text-lg">Audio Generado</h3>
                        <p className="text-xs text-[#1a1a1a]/40 mt-0.5">
                          {mode === 'coqui' ? `XTTS v2 · ${coquiCurrentSpeaker?.gender === 'F' ? '♀' : '♂'} ${coquiSpeaker} · ${coquiCurrentLang?.flag} ${coquiCurrentLang?.name}` :
                            mode === 'webspeech' ? `Web Speech · ${wsvVoices.find(v => v.id === wsvVoice)?.name ?? ''}` :
                              `Kokoro · ${kokoroVoice}`}
                        </p>
                      </div>
                      {mode !== 'webspeech' && (
                        <button onClick={handleDownload}
                          className="flex items-center gap-1.5 px-4 py-2 bg-[#5A5A40] text-white rounded-full text-sm font-medium hover:bg-[#4a4a34] transition-colors">
                          <Download className="w-4 h-4" />WAV
                        </button>
                      )}
                    </div>
                    <audio src={audioUrl} controls className="w-full accent-[#5A5A40]"
                      onPlay={e => startTimeKaraoke(e.currentTarget)}
                      onPause={() => { cancelAnimationFrame(rafRef.current); setActiveChar(-1); setIsKaraoke(false); }}
                      onEnded={() => { cancelAnimationFrame(rafRef.current); setActiveChar(-1); setIsKaraoke(false); }} />
                  </div>
                ) : null}
              </motion.section>
            )}
          </AnimatePresence>
        </div>

        <footer className="mt-16 text-center text-xs text-[#1a1a1a]/35">
          Voice Studio · <a href="https://github.com/coqui-ai/TTS" target="_blank" rel="noopener noreferrer" className="hover:text-[#5A5A40]">Coqui TTS</a> · <a href="https://github.com/hexgrad/kokoro" target="_blank" rel="noopener noreferrer" className="hover:text-[#5A5A40]">Kokoro TTS</a> · Open Source
        </footer>
      </div>
    </div>
  );
}
