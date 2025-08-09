import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { DEFAULTS } from './providers.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.join(__dirname, '..');
const configDir = path.join(rootDir, 'config');

const defaultSystem = 'You are a helpful assistant. Be concise and direct.';

function readJsonSafe(fileName) {
  const filePath = path.join(configDir, fileName);
  if (!fs.existsSync(filePath)) return null;
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (_err) {
    return null;
  }
}

export function loadProviderConfig() {
  const base = {
    system: defaultSystem,
    temperature: DEFAULTS.temperature,
    maxTokens: DEFAULTS.maxTokens,
    timeoutMs: DEFAULTS.timeoutMs,
  };
  return {
    openai: { ...base, ...(readJsonSafe('openai.json') || {}) },
    claude: { ...base, ...(readJsonSafe('claude.json') || {}) },
    gemini: { ...base, ...(readJsonSafe('gemini.json') || {}) },
  };
}

export function loadModelSpecificConfig(model) {
  const base = {
    system: defaultSystem,
    temperature: DEFAULTS.temperature,
    maxTokens: DEFAULTS.maxTokens,
    timeoutMs: DEFAULTS.timeoutMs,
  };
  
  // Check for model-specific config first
  if (model === 'gpt-5-mini' || model === 'gpt-5-mini-2025-08-07') {
    const gpt5MiniConfig = readJsonSafe('gpt5-mini.json');
    if (gpt5MiniConfig) {
      return { ...base, ...gpt5MiniConfig };
    }
  }
  
  // Fall back to provider config
  const openaiConfig = readJsonSafe('openai.json');
  return { ...base, ...openaiConfig };
}


