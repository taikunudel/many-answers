## Multi Answers CLI

Query OpenAI, Anthropic Claude, and Google Gemini in parallel from a single CLI.

### Setup

1. Node.js 18+ is required.
2. Install dependencies:

```bash
npm install
```

3. Set API keys (you can omit providers you don't use):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
# or, Gemini also supports GOOGLE_API_KEY
export GOOGLE_API_KEY=...
```

Or create a `.env` file next to `package.json`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
# Either of the following works for Gemini
GEMINI_API_KEY=...
# or
GOOGLE_API_KEY=...
```

### Usage

Ask a question:

```bash
npx multi-answers -p "Explain vector databases in simple terms"
```

Or via npm script:

```bash
npm run ask -- -p "Best way to debounce input in React?"
```

Pipe input:

```bash
echo "List pros/cons of server-side rendering" | node src/index.js
```

Select models and options:

```bash
node src/index.js \
  -p "Summarize this repo" \
  --model-openai gpt-4o-mini \
  --model-claude claude-3-5-sonnet-latest \
  --model-gemini gemini-1.5-pro \
  --temperature 0.2 \
  --max-tokens 800 \
  --timeout 30000
```

Enable/disable providers:

```bash
node src/index.js -p "Write a haiku" --openai false --claude true --gemini true
```

JSON output:

```bash
node src/index.js -p "Compare quicksort and mergesort" --format json
```

### Notes

- Uses official SDKs: `openai`, `@anthropic-ai/sdk`, `@google/generative-ai`.
- Each provider call is wrapped with a per-provider timeout.
- Defaults: OpenAI `gpt-4o-mini`, Claude `claude-3-5-sonnet-latest`, Gemini `gemini-1.5-pro`.
 - The server/UI only invokes providers you explicitly request. Missing API key messages for other providers won't appear.


# many-answers
# many-answers
