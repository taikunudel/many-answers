import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { askOpenAI, askClaude, askGemini, DEFAULTS, withTimeout, nowMs, drawOpenAI } from './providers.js';
import { loadProviderConfig } from './config.js';
import OpenAI from 'openai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));

// Retry helper: run task up to `attempts` times, each attempt capped by `perAttemptTimeoutMs`.
function runWithRetries(label, taskFactory, attempts = 3, perAttemptTimeoutMs = 5000) {
  async function attempt(n) {
    const result = await withTimeout(taskFactory(), perAttemptTimeoutMs, label);
    if (result.ok) return result;
    if (n + 1 >= attempts) return result;
    return attempt(n + 1);
  }
  return attempt(0);
}

// Streaming (real-time) generation for OpenAI via Server-Sent Events
app.post('/api/stream', async (req, res) => {
  try {
    const { prompt, system, model, temperature, maxTokens, maxCompletionTokens, histories, modelId } = req.body || {};
    if (!prompt) return res.status(400).end('Missing prompt');

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      res.write(`data: ${JSON.stringify({ error: 'Missing OPENAI_API_KEY' })}\n\n`);
      return res.end();
    }
    const client = new OpenAI({ apiKey });

    // Build messages with history
    const messages = [];
    if (system) messages.push({ role: 'system', content: system });
    // Use modelId-specific history for streaming
    const history = (typeof modelId === 'string' && Array.isArray(histories?.[modelId])) 
      ? histories[modelId] 
      : (histories?.openai || []);
    if (Array.isArray(history)) {
      for (const m of history) {
        const role = m?.role === 'assistant' ? 'assistant' : m?.role === 'system' ? 'system' : 'user';
        const content = typeof m?.content === 'string' ? m.content : '';
        if (content) messages.push({ role, content });
      }
    }
    messages.push({ role: 'user', content: prompt });

    const startedAt = nowMs();
    const isGpt5 = /gpt-5/i.test(model || '') || /-2025-/.test(model || '');
    const payload = { model: model || DEFAULTS.openaiModel, messages };
    if (!isGpt5 && typeof temperature === 'number') payload.temperature = temperature;
    if (isGpt5) {
      if (typeof (maxCompletionTokens ?? maxTokens) === 'number') payload.max_completion_tokens = maxCompletionTokens ?? maxTokens;
    } else if (typeof (maxTokens ?? maxCompletionTokens) === 'number') {
      payload.max_tokens = maxTokens ?? maxCompletionTokens;
    }

    const stream = await client.chat.completions.create({ ...payload, stream: true });
    let accumulated = '';
    for await (const part of stream) {
      const delta = part?.choices?.[0]?.delta?.content ?? '';
      if (delta) {
        accumulated += delta;
        res.write(`data: ${JSON.stringify({ provider: 'openai', delta })}\n\n`);
      }
    }
    res.write(`data: ${JSON.stringify({ provider: 'openai', done: true, text: accumulated, latencyMs: nowMs() - startedAt })}\n\n`);
    res.end();
  } catch (err) {
    try { res.write(`data: ${JSON.stringify({ error: String(err?.message || err) })}\n\n`); } catch {}
    res.end();
  }
});

app.post('/api/draw', async (req, res) => {
  try {
    const { prompt, size } = req.body || {};
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing prompt' });
    }
    const startedAt = nowMs();
    const dataUrl = await drawOpenAI({ prompt, size });
    return res.json({ ok: true, image: dataUrl, latencyMs: nowMs() - startedAt });
  } catch (err) {
    return res.json({ ok: false, error: String(err?.message || err) });
  }
});

app.post('/api/ask', async (req, res) => {
  const { prompt, system, providers, models, temperature, maxTokens, timeoutMs, history, histories, showReasoning, useModelConfig, modelId } = req.body || {};
  console.log('=== /api/ask DEBUG ===');
  console.log('modelId:', modelId);
  console.log('prompt:', prompt);
  console.log('histories keys:', Object.keys(histories || {}));
  if (histories && modelId) {
    console.log(`histories[${modelId}] length:`, Array.isArray(histories[modelId]) ? histories[modelId].length : 'not array');
    console.log(`histories[${modelId}] content:`, histories[modelId]);
  }
  console.log('=====================');
  
  if (!prompt || typeof prompt !== 'string' || !prompt.trim()) {
    return res.status(400).json({ error: 'Missing prompt' });
  }
  // Only run providers explicitly requested by the client
  const requested = {
    openai: providers?.openai === true,
    claude: providers?.claude === true,
    gemini: providers?.gemini === true,
  };
  const hasKey = {
    openai: Boolean(process.env.OPENAI_API_KEY),
    claude: Boolean(process.env.ANTHROPIC_API_KEY),
    gemini: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
  };
  const enabled = {
    openai: requested.openai && hasKey.openai,
    claude: requested.claude && hasKey.claude,
    gemini: requested.gemini && hasKey.gemini,
  };
  const config = {
    system: (typeof system === 'string' && system.trim().length > 0) ? system : undefined,
    temperature: typeof temperature === 'number' ? temperature : DEFAULTS.temperature,
    maxTokens: typeof maxTokens === 'number' ? maxTokens : DEFAULTS.maxTokens,
    timeoutMs: typeof timeoutMs === 'number' ? timeoutMs : DEFAULTS.timeoutMs,
    models: {
      openai: models?.openai || DEFAULTS.openaiModel,
      claude: models?.claude || DEFAULTS.claudeModel,
      gemini: models?.gemini || DEFAULTS.geminiModel,
    },
    showReasoning: Boolean(showReasoning),
  };

  // Optional: merge per-provider JSON defaults if requested
  let eff = {
    sys: { openai: config.system, claude: config.system, gemini: config.system },
    temp: { openai: config.temperature, claude: config.temperature, gemini: config.temperature },
    maxT: { openai: config.maxTokens, claude: config.maxTokens, gemini: config.maxTokens },
    tmo: { openai: config.timeoutMs, claude: config.timeoutMs, gemini: config.timeoutMs },
  };
  try {
    if (useModelConfig === true) {
      const CFG = loadProviderConfig();
      eff.sys.openai = (config.system !== undefined) ? config.system : CFG.openai.system;
      eff.sys.claude = (config.system !== undefined) ? config.system : CFG.claude.system;
      eff.sys.gemini = (config.system !== undefined) ? config.system : CFG.gemini.system;
      eff.temp.openai = Number.isFinite(temperature) ? config.temperature : CFG.openai.temperature;
      eff.temp.claude = Number.isFinite(temperature) ? config.temperature : CFG.claude.temperature;
      eff.temp.gemini = Number.isFinite(temperature) ? config.temperature : CFG.gemini.temperature;
      eff.maxT.openai = Number.isFinite(maxTokens) ? config.maxTokens : CFG.openai.maxTokens;
      eff.maxT.claude = Number.isFinite(maxTokens) ? config.maxTokens : CFG.claude.maxTokens;
      eff.maxT.gemini = Number.isFinite(maxTokens) ? config.maxTokens : CFG.gemini.maxTokens;
      eff.tmo.openai = Number.isFinite(timeoutMs) ? config.timeoutMs : CFG.openai.timeoutMs;
      eff.tmo.claude = Number.isFinite(timeoutMs) ? config.timeoutMs : CFG.claude.timeoutMs;
      eff.tmo.gemini = Number.isFinite(timeoutMs) ? config.timeoutMs : CFG.gemini.timeoutMs;
    }
  } catch {}

  // Provider-specific histories with backward-compat fallback to a single 'history'
  function pickHist(providerKey) {
    if (Array.isArray(histories?.[providerKey])) {
      console.log(`Using histories[${providerKey}] (${histories[providerKey].length} items)`);
      return histories[providerKey];
    }
    // If client specified modelId (model1|model2|model3), prefer that card's history
    if (typeof modelId === 'string' && Array.isArray(histories?.[modelId])) {
      console.log(`Using histories[${modelId}] (${histories[modelId].length} items)`);
      return histories[modelId];
    }
    // naive mapping: concatenate all model histories for provider
    if (Array.isArray(histories?.model1) || Array.isArray(histories?.model2) || Array.isArray(histories?.model3)) {
      const combined = [];
      ['model1','model2','model3'].forEach(k=>{if(Array.isArray(histories?.[k])) combined.push(...histories[k]);});
      console.log(`Using combined history (${combined.length} items)`);
      return combined;
    }
    console.log('Using fallback history');
    return history ?? [];
  }
  const historyOpenAI = pickHist('openai');
  const historyClaude = pickHist('claude');
  const historyGemini = pickHist('gemini');

  const jobs = [];
  const results = {};

  if (!hasKey.openai && requested.openai) {
    console.warn('OpenAI disabled: missing OPENAI_API_KEY');
    results.openai = {
      provider: 'openai',
      model: config.models.openai,
      ok: false,
      error: 'Missing OPENAI_API_KEY',
      latencyMs: 0,
    };
  } else if (enabled.openai) {
    const startedAt = nowMs();
    jobs.push(
      runWithRetries('OpenAI', () => (
        askOpenAI({
          prompt,
          system: eff.sys.openai,
          model: config.models.openai,
          temperature: eff.temp.openai,
          maxTokens: eff.maxT.openai,
          history: historyOpenAI,
          useModelConfig: useModelConfig === true,
        })
      ), 3, eff.tmo.openai).then((r) => {
        results.openai = {
          provider: 'openai',
          model: config.models.openai,
          ok: r.ok,
          text: r.ok ? r.value : undefined,
          error: r.ok ? undefined : String(r.error?.message || r.error),
          latencyMs: nowMs() - startedAt,
        };
      })
    );
  }
  if (!hasKey.claude && requested.claude) {
    console.warn('Claude disabled: missing ANTHROPIC_API_KEY');
    results.claude = {
      provider: 'claude',
      model: config.models.claude,
      ok: false,
      error: 'Missing ANTHROPIC_API_KEY',
      latencyMs: 0,
    };
  } else if (enabled.claude) {
    const startedAt = nowMs();
    jobs.push(
      runWithRetries('Claude', () => (
        askClaude({
          prompt,
          system: eff.sys.claude,
          model: config.models.claude,
          temperature: eff.temp.claude,
          maxTokens: eff.maxT.claude,
          history: historyClaude,
        })
      ), 3, eff.tmo.claude).then((r) => {
        results.claude = {
          provider: 'claude',
          model: config.models.claude,
          ok: r.ok,
          text: r.ok ? r.value : undefined,
          error: r.ok ? undefined : String(r.error?.message || r.error),
          latencyMs: nowMs() - startedAt,
        };
      })
    );
  }
  if (!hasKey.gemini && requested.gemini) {
    console.warn('Gemini disabled: missing GEMINI_API_KEY');
    results.gemini = {
      provider: 'gemini',
      model: config.models.gemini,
      ok: false,
      error: 'Missing GEMINI_API_KEY or GOOGLE_API_KEY',
      latencyMs: 0,
    };
  } else if (enabled.gemini) {
    const startedAt = nowMs();
    jobs.push(
      runWithRetries('Gemini', () => (
        askGemini({
          prompt,
          system: eff.sys.gemini,
          model: config.models.gemini,
          temperature: eff.temp.gemini,
          maxTokens: eff.maxT.gemini,
          history: historyGemini,
        })
      ), 3, eff.tmo.gemini).then((r) => {
        results.gemini = {
          provider: 'gemini',
          model: config.models.gemini,
          ok: r.ok,
          text: r.ok ? r.value : undefined,
          error: r.ok ? undefined : String(r.error?.message || r.error),
          latencyMs: nowMs() - startedAt,
        };
      })
    );
  }

  await Promise.all(jobs);
  res.json(results);
});

// Report which providers are currently enabled via env keys
app.get('/api/providers', (req, res) => {
  res.json({
    openai: Boolean(process.env.OPENAI_API_KEY),
    claude: Boolean(process.env.ANTHROPIC_API_KEY),
    gemini: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
  });
});

const basePort = Number(process.env.PORT || 3000);
function startServer(port, retries = 5) {
  const server = app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
    console.log('Env keys detected:', {
      openai: Boolean(process.env.OPENAI_API_KEY),
      claude: Boolean(process.env.ANTHROPIC_API_KEY),
      gemini: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
    });
  });
  server.on('error', (err) => {
    if (err && err.code === 'EADDRINUSE' && retries > 0) {
      console.warn(`Port ${port} is in use. Trying ${port + 1}...`);
      startServer(port + 1, retries - 1);
    } else {
      console.error('Failed to start server:', err);
      process.exit(1);
    }
  });
}
startServer(basePort);


