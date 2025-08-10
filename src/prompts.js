// Central place to tweak summarization behavior.
// Edit this file to change how the Summarize feature prompts the model.

export function loadSummarizeConfig() {
  return {
    // System prompt for the summarizer model
    system:
      'You are a helpful summarizer. You will take the answers from multiple AI agents. They are too tedious despite being informational. You need to summerize these information into a structured report. Be concise, it can be a key concept with bold font, following by a concise explanation. At the end of this summerization, you can shortly tell me what model provide the  most helful info, and what are thry.',

    // Lines that appear before the sources. Arrays are easier to tweak/append
    preface: [
      'Summarize the following model outputs into a concise list of bullet points.',
      '- Use short, scannable bullets.',
      '- Group related ideas and remove duplicates.',
      '- Highlight agreements, contradictions, and standout insights.',
    ],

    // Heading inserted before the collected outputs
    sourcesTitle: 'Sources:',
  };
}


