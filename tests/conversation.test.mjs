import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../frontend/conversation.js',import.meta.url),'utf8');
const {buildConversationPrompt}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

test('preserves recent user and model turns in order',()=>{
 const prompt=buildConversationPrompt([{role:'user',text:'My name is Sam.'},{role:'model',text:'Hi Sam!\n'}],'What is my name?','You are Node.');
 assert.equal(prompt,'You are Node.\nUser: My name is Sam.\nNode: Hi Sam!\nUser: What is my name?\nNode:');
});
test('excludes workspace notices from conversation context',()=>{
 const prompt=buildConversationPrompt([{role:'notice',text:'Backend offline'}],'hi','You are Node.');
 assert.ok(!prompt.includes('Backend offline'));
});
test('caps oversized requests while retaining the newest input',()=>{
 const prompt=buildConversationPrompt([],'hi','x'.repeat(500000));
 assert.equal(prompt.length,4000);
 assert.ok(prompt.endsWith('User: hi\nNode:'));
});
