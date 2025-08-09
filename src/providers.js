import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const DEFAULTS = {
  // Flagship defaults
  openaiModel: 'gpt-5-mini-2025-08-07',
  claudeModel: 'claude-3-5-sonnet-latest',
  geminiModel: 'gemini-1.5-flash',
  temperature: 0.2,
  maxTokens: 1024,
  timeoutMs: 30000,
};

export function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      resolve({ ok: false, error: new Error(`${label} timed out after ${timeoutMs}ms`) });
    }, timeoutMs);
    promise
      .then((value) => {
        clearTimeout(timer);
        resolve({ ok: true, value });
      })
      .catch((error) => {
        clearTimeout(timer);
        resolve({ ok: false, error });
      });
  });
}

export function nowMs() {
  return Date.now();
}

export async function drawOpenAI({ prompt, size = '1024x1024' }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('Missing OPENAI_API_KEY');
  const client = new OpenAI({ apiKey });
  const response = await client.images.generate({
    model: 'gpt-image-1',
    prompt,
    size
  });
  const b64 = response.data?.[0]?.b64_json;
  if (!b64) throw new Error('No image generated');
  return `data:image/png;base64,${b64}`;
}

export async function askOpenAI({ prompt, system, model, temperature, maxTokens, maxCompletionTokens, history, useModelConfig = false }) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('Missing OPENAI_API_KEY');
  const client = new OpenAI({ apiKey });
  
  // Use model-specific config if requested and available
  if (useModelConfig) {
    const { loadModelSpecificConfig } = await import('./config.js');
    const modelConfig = loadModelSpecificConfig(model);
    system = system || modelConfig.system;
    temperature = temperature !== undefined ? temperature : modelConfig.temperature;
    maxTokens = maxTokens !== undefined ? maxTokens : modelConfig.maxTokens;
    maxCompletionTokens = maxCompletionTokens !== undefined ? maxCompletionTokens : modelConfig.maxTokens;
  }

  const messages = [];
  if (system) messages.push({ role: 'system', content: system });
  if (Array.isArray(history)) {
    for (const m of history) {
      const role = m?.role === 'assistant' ? 'assistant' : m?.role === 'system' ? 'system' : 'user';
      const content = typeof m?.content === 'string' ? m.content : '';
      if (content) messages.push({ role, content });
    }
  }
  messages.push({ role: 'user', content: prompt });

  const isGpt5 = /gpt-5/i.test(model) || /-2025-/.test(model);
  const payload = {
    model,
    messages,
  };
  // Only include temperature when the model supports tuning it. Some newer models
  // only allow the default (1) and reject custom values.
  if (!isGpt5 && typeof temperature === 'number') {
    payload.temperature = temperature;
  }
  if (isGpt5) {
    // Newer models expect max_completion_tokens
    if (typeof (maxCompletionTokens ?? maxTokens) === 'number') {
      payload.max_completion_tokens = maxCompletionTokens ?? maxTokens;
    }
  } else {
    if (typeof (maxTokens ?? maxCompletionTokens) === 'number') {
      payload.max_tokens = maxTokens ?? maxCompletionTokens;
    }
  }

  const completion = await client.chat.completions.create(payload);
  const choice = completion.choices?.[0]?.message;
  const content = Array.isArray(choice?.content)
    ? choice.content.map((c) => (typeof c === 'string' ? c : c?.text || '')).join('')
    : (choice?.content ?? '');
  return content;
}

export async function askClaude({ prompt, system, model, temperature, maxTokens, history }) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('Missing ANTHROPIC_API_KEY');
  const anthropic = new Anthropic({ apiKey });

  const messages = [];
  if (Array.isArray(history)) {
    for (const m of history) {
      const role = m?.role === 'assistant' ? 'assistant' : 'user';
      const content = typeof m?.content === 'string' ? m.content : '';
      if (content) messages.push({ role, content });
    }
  }
  messages.push({ role: 'user', content: prompt });

  const response = await anthropic.messages.create({
    model,
    system: system || undefined,
    max_tokens: maxTokens,
    temperature,
    messages,
  });
  const text = (response.content || [])
    .map((b) => (b.type === 'text' ? b.text : ''))
    .join('');
  return text;
}

export async function askGemini({ prompt, system, model, temperature, maxTokens, history }) {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error('Missing GEMINI_API_KEY or GOOGLE_API_KEY');
  const genAI = new GoogleGenerativeAI(apiKey);
  const geminiModel = genAI.getGenerativeModel({
    model,
    ...(system ? { systemInstruction: system } : {}),
  });
  const generationConfig = {
    temperature,
    maxOutputTokens: maxTokens,
  };
  const contents = [];
  if (Array.isArray(history)) {
    for (const m of history) {
      const role = m?.role === 'assistant' ? 'model' : 'user';
      const text = typeof m?.content === 'string' ? m.content : '';
      if (text) contents.push({ role, parts: [{ text }] });
    }
  }
  contents.push({ role: 'user', parts: [{ text: prompt }] });

  const result = await geminiModel.generateContent({
    contents,
    generationConfig,
  });
  const text = (await result.response).text();
  return text;
}


