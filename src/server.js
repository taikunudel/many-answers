import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { askOpenAI, askClaude, askGemini, DEFAULTS, withTimeout, nowMs, drawOpenAI } from './providers.js';
import OpenAI from 'openai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, '..', 'public')));

// Streaming (real-time) generation for OpenAI via Server-Sent Events
app.post('/api/stream', async (req, res) => {
  try {
    const { prompt, system, model, temperature, maxTokens, maxCompletionTokens, histories } = req.body || {};
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
    const history = histories?.openai || [];
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
    const payload = { model: model || DEFAULTS.openaiModel, messages, temperature };
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
  const { prompt, system, providers, models, temperature, maxTokens, timeoutMs, history, histories, showReasoning } = req.body || {};
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
    system: system || undefined,
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

  // Provider-specific histories with backward-compat fallback to a single 'history'
  const historyOpenAI = histories?.openai ?? history ?? [];
  const historyClaude = histories?.claude ?? history ?? [];
  const historyGemini = histories?.gemini ?? history ?? [];

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
      withTimeout(
        askOpenAI({ prompt, system: config.system, model: config.models.openai, temperature: config.temperature, maxTokens: config.maxTokens, history: historyOpenAI }),
        config.timeoutMs,
        'OpenAI'
      ).then((r) => {
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
      withTimeout(
        askClaude({ prompt, system: config.system, model: config.models.claude, temperature: config.temperature, maxTokens: config.maxTokens, history: historyClaude }),
        config.timeoutMs,
        'Claude'
      ).then((r) => {
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
      withTimeout(
        askGemini({ prompt, system: config.system, model: config.models.gemini, temperature: config.temperature, maxTokens: config.maxTokens, history: historyGemini }),
        config.timeoutMs,
        'Gemini'
      ).then((r) => {
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

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
  console.log('Env keys detected:', {
    openai: Boolean(process.env.OPENAI_API_KEY),
    claude: Boolean(process.env.ANTHROPIC_API_KEY),
    gemini: Boolean(process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY),
  });
});


