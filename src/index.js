#!/usr/bin/env node
import 'dotenv/config';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import chalk from 'chalk';
import { askOpenAI, askClaude, askGemini, DEFAULTS, withTimeout } from './providers.js';

const DEFAULT_CLI = DEFAULTS;

function nowMs() {
  return Date.now();
}

async function readStdinIfPiped() {
  if (process.stdin.isTTY) return null;
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  const text = Buffer.concat(chunks).toString('utf8').trim();
  return text.length ? text : null;
}




function printDivider() {
  console.log(chalk.gray('────────────────────────────────────────'));
}

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .usage('Usage: $0 -p "your question" [options]')
    .option('prompt', {
      alias: 'p',
      type: 'string',
      describe: 'Prompt to ask all providers',
    })
    .option('system', {
      type: 'string',
      describe: 'System instruction / persona',
    })
    .option('temperature', {
      alias: 't',
      type: 'number',
      default: DEFAULT_CLI.temperature,
    })
    .option('max-tokens', {
      alias: 'm',
      type: 'number',
      default: DEFAULT_CLI.maxTokens,
    })
    .option('timeout', {
      type: 'number',
      default: DEFAULT_CLI.timeoutMs,
      describe: 'Timeout in milliseconds per provider',
    })
    .option('model-openai', {
      type: 'string',
      default: DEFAULT_CLI.openaiModel,
    })
    .option('model-claude', {
      type: 'string',
      default: DEFAULT_CLI.claudeModel,
    })
    .option('model-gemini', {
      type: 'string',
      default: DEFAULT_CLI.geminiModel,
    })
    .option('openai', {
      type: 'boolean',
      default: true,
      describe: 'Enable OpenAI provider',
    })
    .option('claude', {
      type: 'boolean',
      default: true,
      describe: 'Enable Anthropic Claude provider',
    })
    .option('gemini', {
      type: 'boolean',
      default: true,
      describe: 'Enable Google Gemini provider',
    })
    .option('format', {
      alias: 'f',
      type: 'string',
      choices: ['text', 'json'],
      default: 'text',
    })
    .help()
    .parse();

  let prompt = argv.prompt || (argv._.length ? String(argv._.join(' ')).trim() : '');
  if (!prompt) {
    const piped = await readStdinIfPiped();
    if (piped) prompt = piped;
  }
  if (!prompt) {
    console.error(chalk.red('No prompt provided. Use -p "your question" or pipe text.'));
    process.exit(1);
  }

  const config = {
    temperature: argv.temperature,
    maxTokens: argv['max-tokens'],
    timeoutMs: argv.timeout,
    system: argv.system || undefined,
  };

  const jobs = [];
  const results = {};

  if (argv.openai && process.env.OPENAI_API_KEY) {
    const startedAt = nowMs();
    const p = withTimeout(
      askOpenAI({ prompt, system: config.system, model: argv['model-openai'], temperature: config.temperature, maxTokens: config.maxTokens }),
      config.timeoutMs,
      'OpenAI'
    ).then((r) => {
      results.openai = {
        provider: 'openai',
        model: argv['model-openai'],
        ok: r.ok,
        text: r.ok ? r.value : undefined,
        error: r.ok ? undefined : String(r.error?.message || r.error),
        latencyMs: nowMs() - startedAt,
      };
    });
    jobs.push(p);
  }

  if (argv.claude && process.env.ANTHROPIC_API_KEY) {
    const startedAt = nowMs();
    const p = withTimeout(
      askClaude({ prompt, system: config.system, model: argv['model-claude'], temperature: config.temperature, maxTokens: config.maxTokens }),
      config.timeoutMs,
      'Claude'
    ).then((r) => {
      results.claude = {
        provider: 'claude',
        model: argv['model-claude'],
        ok: r.ok,
        text: r.ok ? r.value : undefined,
        error: r.ok ? undefined : String(r.error?.message || r.error),
        latencyMs: nowMs() - startedAt,
      };
    });
    jobs.push(p);
  }

  if (argv.gemini && (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY)) {
    const startedAt = nowMs();
    const p = withTimeout(
      askGemini({ prompt, system: config.system, model: argv['model-gemini'], temperature: config.temperature, maxTokens: config.maxTokens }),
      config.timeoutMs,
      'Gemini'
    ).then((r) => {
      results.gemini = {
        provider: 'gemini',
        model: argv['model-gemini'],
        ok: r.ok,
        text: r.ok ? r.value : undefined,
        error: r.ok ? undefined : String(r.error?.message || r.error),
        latencyMs: nowMs() - startedAt,
      };
    });
    jobs.push(p);
  }

  await Promise.all(jobs);

  if (argv.format === 'json') {
    console.log(JSON.stringify(results, null, 2));
    return;
  }

  const entries = [
    ['OpenAI', results.openai, chalk.cyan],
    ['Claude', results.claude, chalk.magenta],
    ['Gemini', results.gemini, chalk.yellow],
  ];

  for (const [label, res, color] of entries) {
    if (!res) continue;
    const header = `${label} (${res.model}) ${color(`[${res.latencyMs}ms]`)}`;
    console.log(color.bold(header));
    if (res.ok) {
      console.log(res.text?.trim() || '');
    } else {
      console.log(chalk.red(`Error: ${res.error}`));
    }
    printDivider();
  }
}

main().catch((err) => {
  console.error(chalk.red('Fatal error:'), err?.message || String(err));
  process.exit(1);
});


