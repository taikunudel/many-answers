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

// Meta-instructions for OpenAI deep-research models (Responses API)
// Instruct the model to return researcher instructions rather than doing the task.
export const DEEP_RESEARCH_PLANNER_INSTRUCTIONS = `
You will be given a research task by a user. Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or
  dimensions to consider.
- It is of utmost importance that all details from the user are included in
  the instructions.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user
  has not provided them, explicitly state that they are open-ended or default
  to no specific constraint.

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat
  it as flexible or accept all possible options.

4. **Use the First Person**
- Phrase the request from the perspective of the user.

5. **Tables**
- If you determine that including a table will help illustrate, organize, or
  enhance the information in the research output, you must explicitly request
  that the researcher provide them (examples: product comparison, project
  tracking, budget planning, competitor analysis).

6. **Headers and Formatting**
- Include the expected output format in the prompt.
- If the user is asking for content that would be best returned in a
  structured format (e.g. a report, plan, etc.), ask the researcher to format
  as a report with the appropriate headers and formatting that ensures clarity
  and structure.

7. **Language**
- If the user input is in a language other than English, tell the researcher
  to respond in this language, unless the user query explicitly asks for the
  response in a different language.

8. **Sources**
- If specific sources should be prioritized, specify them in the prompt.
- For product and travel research, prefer linking directly to official or
  primary websites.
- For academic or scientific queries, prefer linking directly to the original
  paper or journal publication.
- If the query is in a specific language, prioritize sources published in that
  language.`;

// Simple in-memory API usage log (FIFO)
const USAGE_LOG_MAX = 100;
const usageLog = [];
export function recordUsage(entry) {
  try {
    const enriched = { id: Date.now() + Math.random(), ts: Date.now(), ...entry };
    usageLog.push(enriched);
    while (usageLog.length > USAGE_LOG_MAX) usageLog.shift();
  } catch {}
}
export function getUsageLog() {
  return usageLog.slice().reverse();
}

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

  // Some OpenAI research models (e.g., o4-mini-deep-research-*) only support the Responses API.
  const usesResponsesApi = /deep-research/i.test(model || '');
  if (usesResponsesApi) {
    // Build a single input string from system + history + prompt
    const parts = [];
    // Prepend planner meta-instructions to steer the deep-research model
    parts.push(`System: ${DEEP_RESEARCH_PLANNER_INSTRUCTIONS}`);
    if (system) parts.push(`System: ${system}`);
    if (Array.isArray(history)) {
      for (const m of history) {
        const role = m?.role === 'assistant' ? 'Assistant' : m?.role === 'system' ? 'System' : 'User';
        const content = typeof m?.content === 'string' ? m.content : '';
        if (content) parts.push(`${role}: ${content}`);
      }
    }
    parts.push(`User: ${prompt}`);
    const input = parts.join('\n\n');

    // Deep-research models have strict schemas; include required tools.
    // At least one of web_search_preview or mcp tools must be present.
    const payload = {
      model,
      input,
      tools: [
        { type: 'web_search_preview' },
        { type: 'code_interpreter', container: { type: 'auto' } },
      ],
    };

    recordUsage({ provider: 'openai', endpoint: 'responses.create', model, payload });
    const response = await client.responses.create(payload);
    // SDK exposes convenience field output_text; fall back to stitching output if absent
    if (typeof response?.output_text === 'string' && response.output_text.length > 0) {
      return response.output_text;
    }
    let combined = '';
    const out = response?.output;
    if (Array.isArray(out)) {
      for (const item of out) {
        if (typeof item?.text === 'string') combined += item.text;
        else if (Array.isArray(item?.content)) combined += item.content.map((c) => c?.text || '').join('');
      }
    }
    return combined;
  }

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

  recordUsage({ provider: 'openai', endpoint: 'chat.completions.create', model, payload });
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


