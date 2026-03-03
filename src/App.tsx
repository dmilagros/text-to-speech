/**
 * Voice Studio — Kokoro TTS (inglés) + Web Speech API (español)
 * Audio progresivo en inglés via AudioContext. Español nativo del navegador.
 * Open Source · Apache 2.0 · Sin API key · Sin límites
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import type { WorkerOutMessage, WorkerInMessage } from './tts.worker';
import {
  Volume2,
  Play,
  Square,
  Loader2,
  Mic2,
  Settings2,
  AlertCircle,
  Download,
  Cpu,
  Globe
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// ── Voces inglés (Kokoro TTS) ────────────────────────────────────────────────
const ENGLISH_VOICES = [
  { id: 'af_heart', label: 'Heart', description: 'Femenina americana, cálida ⭐', gender: 'F' },
  { id: 'af_bella', label: 'Bella', description: 'Femenina americana, suave', gender: 'F' },
  { id: 'af_sarah', label: 'Sarah', description: 'Femenina americana, clara', gender: 'F' },
  { id: 'af_nicole', label: 'Nicole', description: 'Femenina americana, suave', gender: 'F' },
  { id: 'af_sky', label: 'Sky', description: 'Femenina americana, ligera', gender: 'F' },
  { id: 'am_adam', label: 'Adam', description: 'Masculina americana, profunda', gender: 'M' },
  { id: 'am_michael', label: 'Michael', description: 'Masculina americana, natural', gender: 'M' },
  { id: 'bf_emma', label: 'Emma', description: 'Femenina británica, elegante', gender: 'F' },
  { id: 'bf_isabella', label: 'Isabella', description: 'Femenina británica, sofist.', gender: 'F' },
  { id: 'bm_george', label: 'George', description: 'Masculina británica, profunda', gender: 'M' },
  { id: 'bm_lewis', label: 'Lewis', description: 'Masculina británica, resonante', gender: 'M' },
];

// ── Dialectos disponibles para el selector de acento ────────────────────────
const ES_DIALECTS = [
  { lang: 'es-US', label: '🇺🇸 Español (EE.UU.)' },
  { lang: 'es-MX', label: '🇲🇽 México' },
  { lang: 'es-CO', label: '🇨🇴 Colombia' },
  { lang: 'es-AR', label: '🇦🇷 Argentina' },
  { lang: 'es-CL', label: '🇨🇱 Chile' },
  { lang: 'es-PE', label: '🇵🇪 Perú' },
  { lang: 'es-VE', label: '🇻🇪 Venezuela' },
  { lang: 'es-ES', label: '🇪🇸 España' },
];

function sortSpanishVoices(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice[] {
  // Google voices primero (mejor calidad), luego el resto
  return [...voices].sort((a, b) => {
    const ag = a.name.toLowerCase().includes('google') ? 0 : 1;
    const bg = b.name.toLowerCase().includes('google') ? 0 : 1;
    return ag - bg;
  });
}

function getDialectLabel(lang: string): string {
  const map: Record<string, string> = {
    'es-MX': '🇲🇽 México',
    'es-CO': '🇨🇴 Colombia',
    'es-AR': '🇦🇷 Argentina',
    'es-US': '🇺🇸 Español US',
    'es-CL': '🇨🇱 Chile',
    'es-PE': '🇵🇪 Perú',
    'es-VE': '🇻🇪 Venezuela',
    'es-ES': '🇪🇸 España',
  };
  for (const [key, label] of Object.entries(map)) {
    if (lang.startsWith(key)) return label;
  }
  return `🌎 ${lang}`;
}

// ── WAV encoder (para download — solo inglés) ─────────────────────────────────
function mergeToWavBlob(chunks: { audio: Float32Array; sampleRate: number }[]): Blob {
  if (chunks.length === 0) return new Blob([], { type: 'audio/wav' });
  const sampleRate = chunks[0].sampleRate;
  const totalLen = chunks.reduce((s, c) => s + c.audio.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const c of chunks) { merged.set(c.audio, offset); offset += c.audio.length; }

  const bpp = 2, ch = 1;
  const dataSize = merged.length * bpp;
  const buf = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buf);
  let p = 0;
  const wS = (s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(p++, s.charCodeAt(i)); };
  const w32 = (d: number) => { v.setUint32(p, d, true); p += 4; };
  const w16 = (d: number) => { v.setUint16(p, d, true); p += 2; };
  wS('RIFF'); w32(36 + dataSize); wS('WAVE');
  wS('fmt '); w32(16); w16(1); w16(ch);
  w32(sampleRate); w32(sampleRate * ch * bpp); w16(ch * bpp); w16(16);
  wS('data'); w32(dataSize);
  for (let i = 0; i < merged.length; i++) {
    const s = Math.max(-1, Math.min(1, merged[i]));
    v.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true); p += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

export default function App() {
  const [text, setText] = useState(
    'Welcome to Voice Studio! This app converts your text into natural-sounding speech using open-source AI models that run entirely in your browser — no API key needed, completely free and unlimited.'
  );
  const [language, setLanguage] = useState<'en' | 'es'>('en');
  const [selectedVoice, setSelectedVoice] = useState('af_heart');

  // Voces en español detectadas del sistema
  const [spanishVoices, setSpanishVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedSpanishVoice, setSelectedSpanishVoice] = useState<string>('');
  // Dialecto (lang) independiente de la voz seleccionada
  const [spanishDialect, setSpanishDialect] = useState<string>('es-US');

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

  const workerRef = useRef<Worker | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextStartRef = useRef<number>(0);
  const receivedChunksRef = useRef<{ audio: Float32Array; sampleRate: number }[]>([]);
  const downloadRef = useRef<string | null>(null);
  const genStartRef = useRef<number>(0);
  const genRateRef = useRef<number>(0);

  // ── Detectar voces en español del sistema ────────────────────────────────
  useEffect(() => {
    const loadVoices = () => {
      const all = window.speechSynthesis.getVoices();
      const esVoices = sortSpanishVoices(all.filter(v => v.lang.startsWith('es')));
      setSpanishVoices(esVoices);
      if (esVoices.length > 0 && !selectedSpanishVoice) {
        setSelectedSpanishVoice(esVoices[0].name);
      }
    };

    loadVoices();
    // En Chrome, las voces se cargan de forma asíncrona
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Inicializar AudioContext (lazy)
  const getAudioCtx = useCallback(() => {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContext();
      nextStartRef.current = 0;
    }
    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  // Reproducir chunk de audio (para inglés/Kokoro)
  const playChunk = useCallback((samples: Float32Array, sampleRate: number) => {
    const ctx = getAudioCtx();
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    const now = ctx.currentTime;
    const start = Math.max(nextStartRef.current, now + 0.02);
    src.start(start);
    nextStartRef.current = start + buffer.duration;
  }, [getAudioCtx]);

  // Web Worker (solo inglés)
  useEffect(() => {
    const worker = new Worker(
      new URL('./tts.worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;

      switch (msg.type) {
        case 'model_loading':
          setStatus('loading');
          setStatusMsg(msg.message);
          setDownloadPct(0);
          setDownloadMB('');
          break;

        case 'download_progress': {
          const pct = Math.round(msg.progress);
          const loadedMB = (msg.loaded / 1024 / 1024).toFixed(1);
          const totalMB = msg.total > 0 ? (msg.total / 1024 / 1024).toFixed(0) : '?';
          setDownloadPct(pct);
          setDownloadMB(`${loadedMB} / ${totalMB} MB`);
          break;
        }

        case 'model_ready':
          setDeviceInfo(msg.device === 'webgpu' ? '⚡ GPU activa' : '🖥 CPU (WASM)');
          setStatus('generating');
          setStatusMsg('');
          setDownloadPct(0);
          setDownloadMB('');
          genStartRef.current = Date.now();
          break;

        case 'chunk': {
          playChunk(msg.audio, msg.sampleRate);
          receivedChunksRef.current.push({ audio: msg.audio, sampleRate: msg.sampleRate });
          const n = receivedChunksRef.current.length;
          setChunksReceived(n);
          const elapsed = (Date.now() - genStartRef.current) / 1000;
          const rate = elapsed > 0 ? n / elapsed : 0;
          genRateRef.current = rate;
          if (rate > 0) {
            setEstimatedTotal(prev => {
              const total = prev > 0 ? prev : n;
              const remaining = Math.max(0, total - n);
              const secsLeft = remaining / rate;
              if (secsLeft < 5) {
                setEta('casi listo...');
              } else {
                const m = Math.floor(secsLeft / 60);
                const s = Math.round(secsLeft % 60);
                setEta(m > 0 ? `~${m} min ${s} seg restantes` : `~${s} seg restantes`);
              }
              return prev;
            });
          }
          if (msg.index === 0) setStatus('playing');
          break;
        }

        case 'done': {
          const blob = mergeToWavBlob(receivedChunksRef.current);
          const url = URL.createObjectURL(blob);
          if (downloadRef.current) URL.revokeObjectURL(downloadRef.current);
          downloadRef.current = url;
          setAudioUrl(url);
          setStatus('idle');
          setStatusMsg('');
          break;
        }

        case 'error':
          setStatus('idle');
          setStatusMsg('');
          setError(msg.message);
          break;
      }
    };

    workerRef.current = worker;
    return () => { worker.terminate(); };
  }, [playChunk]);

  // ── Generar voz en español con Web Speech API ────────────────────────────
  const generateSpanish = useCallback(() => {
    if (!text.trim()) return;

    // Cancelar síntesis anterior si existe
    window.speechSynthesis.cancel();

    setStatus('playing');
    setError(null);
    setAudioUrl(null);
    setChunksReceived(0);

    const utterance = new SpeechSynthesisUtterance(text);
    // Usamos el dialecto seleccionado (acento) independiente de la voz (calidad)
    utterance.lang = spanishDialect;
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    // Asignar la voz seleccionada
    if (selectedSpanishVoice) {
      const voice = spanishVoices.find(v => v.name === selectedSpanishVoice);
      if (voice) utterance.voice = voice;
    }

    utterance.onstart = () => {
      setStatusMsg('Reproduciendo...');
    };

    utterance.onend = () => {
      setStatus('idle');
      setStatusMsg('');
    };

    utterance.onerror = (e) => {
      if (e.error === 'interrupted' || e.error === 'canceled') return;
      setStatus('idle');
      setStatusMsg('');
      setError(`Error de síntesis: ${e.error}`);
    };

    window.speechSynthesis.speak(utterance);
  }, [text, selectedSpanishVoice, spanishVoices, spanishDialect]);

  const generateSpeech = useCallback(() => {
    if (language === 'es') {
      generateSpanish();
      return;
    }

    if (!text.trim() || !workerRef.current) return;

    const est = Math.max(1, Math.ceil(text.length / 120));
    setEstimatedTotal(est);
    setEta('');
    setStatus('loading');
    setError(null);
    setStatusMsg('Iniciando...');
    setChunksReceived(0);
    receivedChunksRef.current = [];
    setAudioUrl(null);

    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    nextStartRef.current = 0;

    const msg: WorkerInMessage = { type: 'generate', text, voice: selectedVoice };
    workerRef.current.postMessage(msg);
  }, [text, language, selectedVoice, generateSpanish]);

  const cancelGeneration = useCallback(() => {
    if (language === 'es') {
      window.speechSynthesis.cancel();
      setStatus('idle');
      setStatusMsg('');
      return;
    }
    workerRef.current?.postMessage({ type: 'cancel' } satisfies WorkerInMessage);
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    setStatus('idle');
    setStatusMsg('');
    setChunksReceived(0);
  }, [language]);

  const handleDownload = () => {
    const url = downloadRef.current || audioUrl;
    if (!url) return;
    const a = document.createElement('a');
    a.href = url;
    a.download = `voice-studio-${selectedVoice}-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const isActive = status !== 'idle';
  const femaleVoices = ENGLISH_VOICES.filter(v => v.gender === 'F');
  const maleVoices = ENGLISH_VOICES.filter(v => v.gender === 'M');
  const femaleSpanish = spanishVoices.filter(v => v.name.toLowerCase().includes('female') || v.name.toLowerCase().includes('mujer') || /[^a-z](f|femenin)/i.test(v.name) || v.gender === 'female');
  const maleSpanish = spanishVoices.filter(v => !femaleSpanish.includes(v));

  return (
    <div className="min-h-screen bg-[#f5f2ed] text-[#1a1a1a] font-sans selection:bg-[#5A5A40]/20">
      <div className="max-w-4xl mx-auto px-6 py-12 md:py-20">

        {/* Header */}
        <header className="mb-12 text-center">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center justify-center p-3 bg-white rounded-2xl shadow-sm border border-black/5 mb-6">
            <Volume2 className="w-8 h-8 text-[#5A5A40]" />
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="text-4xl md:text-5xl font-serif font-light mb-4 tracking-tight">
            Voice Studio
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="text-lg text-[#1a1a1a]/60 max-w-xl mx-auto">
            Síntesis de voz natural con IA open-source.
            Sin límites, sin API key — corre en tu dispositivo.
          </motion.p>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
            className="mt-3 inline-flex items-center gap-2 text-xs text-[#5A5A40]/70 bg-[#5A5A40]/8 px-3 py-1.5 rounded-full">
            <Cpu className="w-3 h-3" />
            {language === 'en' ? 'Kokoro TTS · Apache 2.0' : `Web Speech API · ${spanishVoices.length} voces detectadas`}
            {deviceInfo && language === 'en' && <span className="font-medium text-[#5A5A40]">· {deviceInfo}</span>}
          </motion.div>
        </header>

        <main className="grid gap-8">

          {/* Idioma */}
          <motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
            className="bg-white rounded-[32px] p-6 shadow-sm border border-black/5">
            <div className="flex items-center gap-2 mb-4 text-[#5A5A40]">
              <Globe className="w-5 h-5" />
              <h2 className="text-sm font-semibold uppercase tracking-widest">Idioma</h2>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {([
                { id: 'en', flag: '🇺🇸', label: 'English', sub: 'Kokoro TTS · Grado A · Streaming' },
                { id: 'es', flag: '🌎', label: 'Español Latino', sub: `Web Speech API · ${spanishVoices.length} voces` },
              ] as const).map(({ id, flag, label, sub }) => (
                <button key={id} onClick={() => setLanguage(id)}
                  className={`flex flex-col items-start p-4 rounded-2xl border transition-all text-left ${language === id ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow-md' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a] hover:border-[#5A5A40]/30'}`}>
                  <span className="font-medium">{flag} {label}</span>
                  <span className={`text-xs mt-1 ${language === id ? 'text-white/70' : 'text-[#1a1a1a]/50'}`}>{sub}</span>
                </button>
              ))}
            </div>
          </motion.section>

          {/* Texto */}
          <motion.section initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}
            className="bg-white rounded-[32px] p-8 shadow-sm border border-black/5">
            <div className="flex items-center gap-2 mb-4 text-[#5A5A40]">
              <Mic2 className="w-5 h-5" />
              <h2 className="text-sm font-semibold uppercase tracking-widest">Texto</h2>
            </div>
            <textarea value={text} onChange={(e) => setText(e.target.value)}
              placeholder={language === 'es' ? 'Escribe el texto en español...' : 'Enter the English text...'}
              className="w-full h-48 p-4 bg-[#f9f8f6] rounded-2xl border border-black/5 focus:outline-none focus:ring-2 focus:ring-[#5A5A40]/20 transition-all resize-none text-lg leading-relaxed" />
            <div className="mt-2 text-right text-xs text-[#1a1a1a]/40">
              {text.length.toLocaleString()} caracteres
            </div>
          </motion.section>

          {/* Controles */}
          <div className="grid md:grid-cols-3 gap-8">
            <motion.section initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
              className="md:col-span-2 bg-white rounded-[32px] p-8 shadow-sm border border-black/5">
              <div className="flex items-center gap-2 mb-6 text-[#5A5A40]">
                <Settings2 className="w-5 h-5" />
                <h2 className="text-sm font-semibold uppercase tracking-widest">Seleccionar Voz</h2>
              </div>
              <AnimatePresence mode="wait">
                {language === 'en' ? (
                  <motion.div key="en" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <p className="text-xs font-medium text-[#1a1a1a]/40 uppercase tracking-wider mb-3">Femeninas</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
                      {femaleVoices.map(v => (
                        <button key={v.id} onClick={() => setSelectedVoice(v.id)}
                          className={`flex flex-col items-start p-4 rounded-2xl border transition-all text-left ${selectedVoice === v.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow-md' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a] hover:border-[#5A5A40]/30'}`}>
                          <span className="font-medium mb-1">{v.label}</span>
                          <span className={`text-xs ${selectedVoice === v.id ? 'text-white/70' : 'text-[#1a1a1a]/50'}`}>{v.description}</span>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs font-medium text-[#1a1a1a]/40 uppercase tracking-wider mb-3">Masculinas</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {maleVoices.map(v => (
                        <button key={v.id} onClick={() => setSelectedVoice(v.id)}
                          className={`flex flex-col items-start p-4 rounded-2xl border transition-all text-left ${selectedVoice === v.id ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow-md' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a] hover:border-[#5A5A40]/30'}`}>
                          <span className="font-medium mb-1">{v.label}</span>
                          <span className={`text-xs ${selectedVoice === v.id ? 'text-white/70' : 'text-[#1a1a1a]/50'}`}>{v.description}</span>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="es" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    {spanishVoices.length === 0 ? (
                      <div className="text-center py-8 text-[#1a1a1a]/40">
                        <p className="text-sm">No se detectaron voces en español.</p>
                        <p className="text-xs mt-1">Instala voces en español en la configuración del sistema.</p>
                      </div>
                    ) : (
                      <>
                        {/* Selector de dialecto/acento */}
                        <p className="text-xs font-medium text-[#1a1a1a]/40 uppercase tracking-wider mb-3">Dialecto / Acento</p>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-6">
                          {ES_DIALECTS.map(d => (
                            <button key={d.lang} onClick={() => setSpanishDialect(d.lang)}
                              className={`px-3 py-2 rounded-xl border text-xs font-medium transition-all text-center ${spanishDialect === d.lang ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow-md' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a] hover:border-[#5A5A40]/30'}`}>
                              {d.label}
                            </button>
                          ))}
                        </div>

                        {/* Selector de voz */}
                        <p className="text-xs font-medium text-[#1a1a1a]/40 uppercase tracking-wider mb-3">Voz (Motor)</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {spanishVoices.map(v => {
                            const isGoogle = v.name.toLowerCase().includes('google');
                            return (
                              <button key={v.name} onClick={() => setSelectedSpanishVoice(v.name)}
                                className={`flex flex-col items-start p-4 rounded-2xl border transition-all text-left ${selectedSpanishVoice === v.name ? 'bg-[#5A5A40] border-[#5A5A40] text-white shadow-md' : 'bg-[#f9f8f6] border-black/5 text-[#1a1a1a] hover:border-[#5A5A40]/30'}`}>
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-medium text-sm leading-tight">{v.name}</span>
                                  {isGoogle && (
                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${selectedSpanishVoice === v.name ? 'bg-white/20 text-white' : 'bg-[#5A5A40]/10 text-[#5A5A40]'}`}>
                                      ⭐ HD
                                    </span>
                                  )}
                                </div>
                                <span className={`text-xs ${selectedSpanishVoice === v.name ? 'text-white/70' : 'text-[#1a1a1a]/50'}`}>
                                  {v.lang}{v.localService ? ' · Local' : ' · Online'}
                                </span>
                              </button>
                            );
                          })}
                        </div>
                        <p className="text-xs text-[#1a1a1a]/40 mt-4 italic">
                          💡 Combina una voz ⭐ HD con el dialecto que prefieras para mejor resultado.
                        </p>
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>

            {/* Botón */}
            <motion.section initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}
              className="flex flex-col gap-3">
              <button
                onClick={isActive ? cancelGeneration : generateSpeech}
                disabled={!text.trim() && !isActive}
                className="flex-1 bg-[#5A5A40] hover:bg-[#4a4a34] text-white rounded-[32px] p-8 flex flex-col items-center justify-center gap-3 transition-all shadow-lg active:scale-95 relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {/* Barra de descarga del modelo */}
                {status === 'loading' && downloadPct > 0 && (
                  <motion.div
                    className="absolute bottom-0 left-0 h-1.5 bg-white/40"
                    animate={{ width: `${downloadPct}%` }}
                    transition={{ ease: 'easeOut', duration: 0.3 }}
                  />
                )}
                {/* Barra pulsante durante generación */}
                {(status === 'generating' || status === 'playing') && (
                  <motion.div
                    className="absolute bottom-0 left-0 h-1 bg-white/25"
                    animate={{ width: ['0%', '100%'] }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                  />
                )}

                {isActive
                  ? <Loader2 className="w-10 h-10 animate-spin opacity-90" />
                  : <Play className="w-12 h-12 fill-current" />
                }

                <div className="text-center w-full px-1">
                  <span className="text-base font-serif italic block leading-snug mb-2">
                    {status === 'idle' && 'Generar Voz'}
                    {status === 'loading' && (downloadPct > 0 ? 'Descargando modelo...' : 'Iniciando...')}
                    {status === 'generating' && 'Generando audio...'}
                    {status === 'playing' && (language === 'es' ? 'Reproduciendo...' : 'Generando y reproduciendo...')}
                  </span>

                  {/* Progreso descarga */}
                  {status === 'loading' && downloadPct > 0 && (
                    <div className="w-full">
                      <div className="flex justify-between text-xs text-white/60 mb-1">
                        <span className="font-mono font-bold">{downloadPct}%</span>
                        <span>{downloadMB}</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-1.5">
                        <motion.div
                          className="bg-white/60 h-1.5 rounded-full"
                          animate={{ width: `${downloadPct}%` }}
                          transition={{ ease: 'easeOut', duration: 0.3 }}
                        />
                      </div>
                      <p className="text-xs text-white/35 mt-1.5">Solo se descarga la primera vez</p>
                    </div>
                  )}

                  {/* Contador oraciones (inglés) */}
                  {language === 'en' && (status === 'generating' || status === 'playing') && chunksReceived > 0 && (
                    <div className="text-xs text-white/60 space-y-0.5">
                      <p>
                        {chunksReceived} / ~{estimatedTotal} {estimatedTotal === 1 ? 'oración' : 'oraciones'}
                      </p>
                      {eta && <p className="font-medium text-white/80">{eta}</p>}
                    </div>
                  )}

                  {isActive && <p className="text-xs text-white/30 mt-1">Toca para cancelar</p>}
                </div>

                {isActive && <Square className="w-3.5 h-3.5 absolute top-4 right-4 opacity-40" />}
              </button>
            </motion.section>
          </div>

          {/* Output: WAV solo para inglés */}
          <AnimatePresence>
            {(audioUrl || error) && (
              <motion.section
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }}
                className={`rounded-[32px] p-8 border ${error ? 'bg-red-50 border-red-100' : 'bg-white border-black/5 shadow-sm'}`}
              >
                {error ? (
                  <div className="flex items-center gap-3 text-red-600">
                    <AlertCircle className="w-6 h-6 flex-shrink-0" />
                    <p className="font-medium">{error}</p>
                  </div>
                ) : (
                  <div className="w-full">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-serif italic text-xl">Audio Completo</h3>
                        <p className="text-xs text-[#1a1a1a]/40 mt-0.5">Listo para reproducir y descargar</p>
                      </div>
                      <button onClick={handleDownload}
                        className="flex items-center gap-1.5 px-4 py-2 bg-[#5A5A40] text-white rounded-full text-sm font-medium hover:bg-[#4a4a34] transition-colors">
                        <Download className="w-4 h-4" />
                        Descargar WAV
                      </button>
                    </div>
                    <audio src={audioUrl!} controls className="w-full h-12 accent-[#5A5A40]" />
                  </div>
                )}
              </motion.section>
            )}
          </AnimatePresence>
        </main>

        <footer className="mt-20 pt-8 border-t border-black/5 text-center text-[#1a1a1a]/40 text-sm">
          <p>
            © {new Date().getFullYear()} Voice Studio ·{' '}
            <a href="https://github.com/hexgrad/kokoro" target="_blank" rel="noopener noreferrer" className="hover:text-[#5A5A40] transition-colors">Kokoro TTS</a>
            {' · '}
            <a href="https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API" target="_blank" rel="noopener noreferrer" className="hover:text-[#5A5A40] transition-colors">Web Speech API</a>
            {' · '}Open Source · Apache 2.0
          </p>
        </footer>
      </div>
    </div>
  );
}
