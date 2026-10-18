# comfyui-agent-kit

把一个**全局 ComfyUI MCP server** 同时接入 **Codex / Claude Code / ZCode / DeepSeek Harness (DSH)**。
同一个 `comfyui-local` server、同一组环境变量、同一套 18 个工具；四个 agent 之间互不干扰。

> **English**: One local MCP server that exposes ComfyUI's HTTP API plus SDXL LoRA
> training as 18 tools, with registration snippets for Codex, Claude Code, ZCode and
> DeepSeek Harness. The DSH side ships a Cordis plugin that turns the same tools into
> native DSH tools.

---

## 为什么需要 DSH 插件

Codex / Claude Code / ZCode 都有自己的 MCP 客户端，注册一下就能直连 `server.py`。
DSH 的工具模型是 Cordis，**没有外部 MCP 挂载机制**，所以这里额外提供一个薄桥接插件
`dsh-plugin/`：启动时读一次 `tools/list`，把 18 个工具用 `ctx.tools.register` 注册成
一等原生工具；每次调用再拉起短命 `server.py` 转发。

```
                    ┌─ Codex        （原生 MCP，TOML）
server.py  ◄────────┼─ Claude Code  （原生 MCP，JSON）
features.py         ├─ ZCode        （原生 MCP，JSON）
   │ 18 tools       └─ DSH          （Cordis 插件 dsh-plugin/）
   ▼
ComfyUI HTTP API (127.0.0.1:8188)
```

桥接插件只读复用 server，**不修改也不复刻**它的实现，更不会碰另外三个 agent 的注册。

## 仓库结构

```
comfyui-agent-kit/
├─ mcp-server/
│  ├─ server.py        # stdio JSON-RPC，8 个基础工具（仅用 Python 标准库）
│  └─ features.py      # 10 个扩展工具（工作流/节点/项目资产/SDXL LoRA 训练）
├─ dsh-plugin/
│  ├─ package.json     # dsh.bundle.patch 指向下面的补丁
│  ├─ cordis.patch.yml # 只插入自己一行，不替换任何官方组件
│  └─ lib/index.js     # 桥接实现
└─ examples/
   └─ sdxl-lora-txt2img.json
```

## 环境要求

| 依赖 | 说明 |
|---|---|
| ComfyUI | 需已启动，默认 `http://127.0.0.1:8188`（HTTP API） |
| Python | 3.8+，**只用标准库**，无需 pip 安装任何包 |
| Node.js | ≥ 22（仅 DSH 插件需要） |
| kohya_ss | 可选，仅 `comfyui_lora_*` 训练工具需要 |

## 安装

### 1. 放置 MCP server

把 `mcp-server/` 两个文件放到任意固定目录。默认查找位置：

```
~/.agents/mcp-servers/comfyui/server.py
```

（Windows 即 `C:\Users\<你>\.agents\mcp-servers\comfyui\server.py`。）

### 2. 各 agent 注册

**Codex** — `~/.codex/config.toml`：

```toml
[mcp_servers.comfyui-local]
command = "py"            # Windows；其他平台用 python3
args = ["-3", "C:\\Users\\<你>\\.agents\\mcp-servers\\comfyui\\server.py"]
startup_timeout_sec = 10

[mcp_servers.comfyui-local.env]
COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_HOME = "E:\\Program\\ComfyUI"
COMFYUI_PROJECT_ROOT = "D:\\comfyui-project"
KOHYA_SS_ROOT = "E:\\Program\\kohya_ss"
```

**Claude Code** — `~/.claude.json`：

```json
{
  "mcpServers": {
    "comfyui-local": {
      "type": "stdio",
      "command": "C:\\Windows\\py.exe",
      "args": ["-3", "C:\\Users\\<你>\\.agents\\mcp-servers\\comfyui\\server.py"],
      "env": {
        "COMFYUI_URL": "http://127.0.0.1:8188",
        "COMFYUI_HOME": "E:\\Program\\ComfyUI",
        "COMFYUI_PROJECT_ROOT": "D:\\comfyui-project",
        "KOHYA_SS_ROOT": "E:\\Program\\kohya_ss"
      }
    }
  }
}
```

**ZCode** — `~/.zcode/cli/config.json`：

```json
{
  "mcp": {
    "servers": {
      "comfyui-local": {
        "command": "C:\\Windows\\py.exe",
        "args": ["-3", "C:\\Users\\<你>\\.agents\\mcp-servers\\comfyui\\server.py"],
        "env": {
          "COMFYUI_URL": "http://127.0.0.1:8188",
          "COMFYUI_HOME": "E:\\Program\\ComfyUI",
          "COMFYUI_PROJECT_ROOT": "D:\\comfyui-project",
          "KOHYA_SS_ROOT": "E:\\Program\\kohya_ss"
        }
      }
    }
  }
}
```

**DSH** — 装本地包并挂进 profile：

```powershell
dsh plugin --profile desktop add "C:\path\to\comfyui-agent-kit\dsh-plugin"
```

然后在该 profile 的 `package.json` 里把 `dsh-comfyui-mcp` 加进 `dsh.profile.bundles`，
重启桌面端：

```json
{
  "dsh": {
    "profile": {
      "bundles": [
        "@deepseek-ai/dsh-web-app",
        "dsh-comfyui-mcp"
      ]
    }
  }
}
```

装好后 `comfyui_*` 会直接出现在 DSH 的工具列表里，**不需要再手写 subprocess 桥接**。

## 环境变量

| 变量 | 默认值 | 用途 |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI HTTP 地址 |
| `COMFYUI_TIMEOUT` | `30` | 单次 HTTP 超时（秒） |
| `COMFYUI_HOME` | `E:\Program\ComfyUI` | ComfyUI 安装目录（定位 workflows / models） |
| `COMFYUI_PROJECT_ROOT` | `~/comfyui-project` | 目标项目根（LoRA 数据集与 `assets/art`） |
| `KOHYA_SS_ROOT` | `E:\Program\kohya_ss` | kohya_ss 安装目录 |
| `COMFYUI_MCP_PYTHON` | `C:\Windows\py.exe`（Windows）/ `python3` | DSH 插件用来拉起 server 的解释器 |
| `COMFYUI_MCP_SERVER` | `~/.agents/mcp-servers/comfyui/server.py` | server 脚本路径 |

后三个只影响 DSH 插件；其余对所有 agent 生效。

## 工具（18 个）

| 组 | 工具 |
|---|---|
| ComfyUI 基础 | `comfyui_status` · `comfyui_list_models` · `comfyui_queue_workflow` · `comfyui_queue_status` · `comfyui_get_history` · `comfyui_interrupt` · `comfyui_upload_image` · `comfyui_download_output` |
| 节点与工作流 | `comfyui_search_nodes` · `comfyui_get_node_schema` · `comfyui_build_workflow` · `comfyui_validate_workflow` · `comfyui_save_workflow` |
| 项目资产 | `comfyui_apply_project_art` |
| SDXL LoRA 训练 | `comfyui_lora_preflight` · `comfyui_lora_prepare` · `comfyui_lora_start` · `comfyui_lora_status` |

## 示例

`examples/sdxl-lora-txt2img.json` 是一个最小 SDXL 文生图 API workflow
（Checkpoint → LoRA → 双 CLIP → KSampler → VAEDecode → SaveImage）：

```jsonc
// 直接提交：comfyui_queue_workflow({"workflow_json": "<文件内容>"})
// 校验：    comfyui_validate_workflow({"workflow_json": "<文件内容>"})
```

把 `lora_name` 换成你自己的 LoRA 相对路径即可，**子目录要带上**，例如
`my_lora\my_lora.safetensors`。

## 安全与隐私

- server 只连 `COMFYUI_URL`，不访问任何外网、无遥测。
- 默认只监听回环地址；如需暴露到局域网请自行加鉴权。
- `comfyui_apply_project_art` 覆盖已存在立绘时要求显式 `overwrite=true`，并保留
  `*.before-comfyui.png` 备份。
- 仓库内不含任何 API key / token。

## License

MIT
