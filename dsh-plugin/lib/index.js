/**
 * dsh-comfyui-mcp — host half.
 *
 * Bridges a global MCP server ("comfyui-local", stdio JSON-RPC) into first-class
 * DSH tools. On start it fetches the server's tool list (with inputSchema) and
 * registers every `comfyui_*` tool through `ctx.tools.register`; each call then
 * spawns a short-lived server process and forwards the request.
 *
 * The MCP server is treated as a read-only dependency: this plugin does not
 * modify it, and it does not touch the Codex / Claude / ZCode registrations of
 * the same server (those agents keep talking to it directly).
 */

import { spawn, spawnSync } from 'node:child_process';
import { homedir } from 'node:os';
import { join } from 'node:path';

export const inject = ['tools'];

/** Python launcher. Override with COMFYUI_MCP_PYTHON. */
const PYTHON =
  process.env.COMFYUI_MCP_PYTHON ||
  (process.platform === 'win32' ? 'C:\\Windows\\py.exe' : 'python3');

/** Path to the comfyui-local MCP server. Override with COMFYUI_MCP_SERVER. */
const SERVER =
  process.env.COMFYUI_MCP_SERVER ||
  join(homedir(), '.agents', 'mcp-servers', 'comfyui', 'server.py');

/**
 * Environment handed to the MCP server. Every entry falls back to the server's
 * own default when unset, so only real overrides are forwarded.
 */
const BASE_ENV = {
  COMFYUI_URL: process.env.COMFYUI_URL || 'http://127.0.0.1:8188',
  ...(process.env.COMFYUI_HOME ? { COMFYUI_HOME: process.env.COMFYUI_HOME } : {}),
  ...(process.env.PET_PROJECT_ROOT
    ? { PET_PROJECT_ROOT: process.env.PET_PROJECT_ROOT }
    : {}),
  ...(process.env.KOHYA_SS_ROOT ? { KOHYA_SS_ROOT: process.env.KOHYA_SS_ROOT } : {}),
  PYTHONIOENCODING: 'utf-8',
};

function rpcEnv() {
  return { ...process.env, ...BASE_ENV };
}

/**
 * Send several JSON-RPC requests in one shot and collect the line-delimited
 * responses. server.py reads stdin line by line and exits when stdin closes.
 */
function rpcLines(requests, timeoutMs = 120_000) {
  const input = requests.map((r) => JSON.stringify(r)).join('\n') + '\n';
  const res = spawnSync(PYTHON, ['-3', SERVER], {
    input,
    encoding: 'utf-8',
    env: rpcEnv(),
    windowsHide: true,
    maxBuffer: 64 * 1024 * 1024,
    timeout: timeoutMs,
  });
  if (res.error) throw res.error;
  const out = [];
  for (const line of (res.stdout || '').split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s)); } catch { /* skip non-JSON lines */ }
  }
  if (!out.length) {
    const err = (res.stderr || '').trim();
    throw new Error('comfyui-local MCP did not respond' + (err ? ': ' + err.slice(-400) : ''));
  }
  return out;
}

function clientInfo() {
  return { name: 'dsh-comfyui-mcp', version: '0.1.1' };
}

function fetchToolList() {
  const msgs = rpcLines([
    { jsonrpc: '2.0', id: 0, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: clientInfo() } },
    { jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} },
  ]);
  const tl = msgs.find((m) => m.id === 1);
  const tools = tl && tl.result && tl.result.tools;
  if (!Array.isArray(tools)) throw new Error('tools/list returned nothing');
  return tools;
}

/** Call one MCP tool: start a short-lived server, send initialize + tools/call. */
function callTool(toolName, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, ['-3', SERVER], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: rpcEnv(),
      windowsHide: true,
    });
    let stdout = '';
    let stderr = '';
    proc.stdout.setEncoding('utf-8');
    proc.stderr.setEncoding('utf-8');
    proc.stdout.on('data', (d) => { stdout += d; });
    proc.stderr.on('data', (d) => { stderr += d; });
    proc.on('error', reject);
    proc.on('close', () => {
      for (const raw of stdout.split('\n')) {
        const s = raw.trim();
        if (!s) continue;
        let msg;
        try { msg = JSON.parse(s); } catch { continue; }
        if (msg.id !== 1) continue;
        if (msg.error) return reject(new Error(JSON.stringify(msg.error)));
        const r = msg.result;
        const text = r && r.content && r.content[0] && r.content[0].text;
        if (r && r.isError) return reject(new Error(text || 'MCP tool returned an error'));
        try { return resolve(text === undefined ? (r || null) : JSON.parse(text)); }
        catch { return resolve(text); }
      }
      reject(new Error('MCP did not respond' + (stderr.trim() ? ': ' + stderr.trim().slice(-300) : '')));
    });
    proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 0, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: clientInfo() } }) + '\n');
    proc.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: toolName, arguments: args || {} } }) + '\n');
    proc.stdin.end();
  });
}

export function apply(ctx) {
  const tools = ctx.tools ?? ctx.get('tools');
  if (!tools || typeof tools.register !== 'function') {
    console.error('[dsh-comfyui-mcp] no tools service; cannot register tools');
    return;
  }
  const disposers = [];
  try {
    const list = fetchToolList();
    for (const t of list) {
      const tname = t && t.name;
      if (typeof tname !== 'string' || !tname.startsWith('comfyui_')) continue;
      const parameters = (t.inputSchema && typeof t.inputSchema === 'object') ? t.inputSchema : { type: 'object', properties: {} };
      const disposer = tools.register({
        name: tname,
        description: (t.description || 'ComfyUI / LoRA training tool (from comfyui-local MCP)'),
        parameters,
        output: {
          schema: { type: 'object', additionalProperties: true },
          render(args, value) {
            const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
            return [{ type: 'text', text }];
          },
        },
        async execute(args) {
          const v = await callTool(tname, args);
          return (v && typeof v === 'object') ? v : { result: v };
        },
      });
      if (typeof disposer === 'function') disposers.push(disposer);
    }
    ctx.effect(() => () => {
      for (const d of disposers) { try { d(); } catch { /* noop */ } }
    }, 'dsh-comfyui-mcp.tool-disposers');
  } catch (err) {
    // A failed registration must not take the host down: log and stay inert.
    console.error('[dsh-comfyui-mcp] tool registration failed:', (err && err.message) || err);
  }
}
