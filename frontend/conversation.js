export function buildConversationPrompt(messages, text, systemPrompt) {
  const turns = messages
    .filter(message => message.role === 'user' || message.role === 'model')
    .map(message => `${message.role === 'user' ? 'User' : 'Node'}: ${message.text.trimEnd()}`);
  turns.push(`User: ${text}`, 'Node:');
  // The worker applies the model's tighter 256-byte context limit.
  // Keep newest turns; never treat workspace notices as model/user dialogue.
  return `${systemPrompt}\n${turns.join('\n')}`.slice(-4000);
}
