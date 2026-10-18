#!/usr/bin/env python3
"""Local MCP bridge for ComfyUI's HTTP API; Python standard library only."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from features import call_tool as call_extended_tool, tools as extended_tools

NAME = "comfyui-local"
VERSION = "1.0.0"
BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
TIMEOUT = float(os.environ.get("COMFYUI_TIMEOUT", "30"))


def request(method: str, path: str, payload: Any = None, timeout: float | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or TIMEOUT) as response:
            raw = response.read()
            if not raw:
                return {"ok": True, "status": response.status}
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type or raw[:1] in (b"{", b"["):
                return json.loads(raw.decode("utf-8"))
            return {"status": response.status, "content_type": content_type, "bytes": len(raw)}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 ComfyUI ({BASE_URL}): {exc.reason}") from exc


def tools() -> list[dict[str, Any]]:
    base = [
        {"name": "comfyui_status", "description": "检查 ComfyUI 是否在线，并返回系统/GPU 状态。", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "comfyui_list_models", "description": "列出 ComfyUI 指定类别的模型文件。", "inputSchema": {"type": "object", "properties": {"category": {"type": "string", "enum": ["checkpoints", "loras", "vae", "controlnet", "upscale_models", "embeddings", "clip", "diffusion_models", "text_encoders"]}}, "required": ["category"]}},
        {"name": "comfyui_queue_workflow", "description": "将 API 格式的 ComfyUI workflow 加入队列。workflow_json 应为节点 ID 到节点对象的 JSON 对象。", "inputSchema": {"type": "object", "properties": {"workflow_json": {"type": "string", "description": "完整 API workflow JSON"}, "client_id": {"type": "string"}}, "required": ["workflow_json"]}},
        {"name": "comfyui_queue_status", "description": "查看 ComfyUI 当前运行任务和等待队列。", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "comfyui_get_history", "description": "按 prompt_id 获取生成任务状态和输出文件信息。", "inputSchema": {"type": "object", "properties": {"prompt_id": {"type": "string"}}, "required": ["prompt_id"]}},
        {"name": "comfyui_interrupt", "description": "中断当前正在执行的 ComfyUI 任务。", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "comfyui_upload_image", "description": "上传本机图片到 ComfyUI input 目录；支持常见 PNG/JPEG/WebP/BMP/GIF。", "inputSchema": {"type": "object", "properties": {"image_path": {"type": "string"}, "subfolder": {"type": "string"}, "overwrite": {"type": "boolean"}}, "required": ["image_path"]}},
        {"name": "comfyui_download_output", "description": "从 ComfyUI 下载一个 output 文件到指定本机路径。", "inputSchema": {"type": "object", "properties": {"filename": {"type": "string"}, "output_path": {"type": "string"}, "subfolder": {"type": "string"}, "type": {"type": "string", "enum": ["output", "temp", "input"]}}, "required": ["filename", "output_path"]}},
    ]
    return base + extended_tools()


def call_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "comfyui_status":
        return {"base_url": BASE_URL, "system_stats": request("GET", "/system_stats"), "queue": request("GET", "/queue")}
    if name == "comfyui_list_models":
        category = args["category"]
        return {"category": category, "models": request("GET", "/models/" + urllib.parse.quote(category, safe=""))}
    if name == "comfyui_queue_workflow":
        workflow = json.loads(args["workflow_json"])
        if not isinstance(workflow, dict):
            raise ValueError("workflow_json 必须是 API workflow JSON 对象")
        body = {"prompt": workflow, "client_id": args.get("client_id") or str(uuid.uuid4())}
        return request("POST", "/prompt", body)
    if name == "comfyui_queue_status":
        return request("GET", "/queue")
    if name == "comfyui_get_history":
        prompt_id = urllib.parse.quote(args["prompt_id"], safe="")
        return request("GET", "/history/" + prompt_id)
    if name == "comfyui_interrupt":
        return request("POST", "/interrupt", {})
    if name == "comfyui_upload_image":
        path = Path(args["image_path"]).expanduser()
        if not path.is_file():
            raise ValueError(f"找不到图片文件: {path}")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            raise ValueError("仅允许上传常见图片格式")
        boundary = "----ComfyUIMCP" + uuid.uuid4().hex
        parts = []
        def field(key: str, value: str) -> None:
            parts.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()])
        field("type", "input")
        field("subfolder", args.get("subfolder", ""))
        field("overwrite", "true" if args.get("overwrite", False) else "false")
        filename = path.name.replace('"', "")
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
        parts.append(path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(BASE_URL + "/upload/image", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI HTTP {exc.code}: {exc.read().decode(errors='replace')[:4000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 ComfyUI ({BASE_URL}): {exc.reason}") from exc
    if name == "comfyui_download_output":
        query = urllib.parse.urlencode({"filename": args["filename"], "subfolder": args.get("subfolder", ""), "type": args.get("type", "output")})
        req = urllib.request.Request(BASE_URL + "/view?" + query)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                content = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"ComfyUI HTTP {exc.code}: {exc.read().decode(errors='replace')[:4000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 ComfyUI ({BASE_URL}): {exc.reason}") from exc
        target = Path(args["output_path"]).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"saved_to": str(target), "bytes": len(content)}
    extended_result = call_extended_tool(name, args, request)
    if extended_result is not None:
        return extended_result
    raise ValueError(f"未知工具: {name}")


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            method = msg.get("method")
            msg_id = msg.get("id")
            if method == "notifications/initialized" or method == "notifications/cancelled":
                continue
            if method == "initialize":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": NAME, "version": VERSION}}})
            elif method == "ping":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            elif method == "tools/list":
                send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools()}})
            elif method == "tools/call":
                params = msg.get("params", {})
                try:
                    result = call_tool(params["name"], params.get("arguments", {}))
                    content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {"content": content, "isError": False}})
                except Exception as exc:
                    send({"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True}})
            elif msg_id is not None:
                send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})
        except Exception as exc:
            print(f"{NAME}: {exc}", file=sys.stderr, flush=True)
            if "msg_id" in locals() and msg_id is not None:
                send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": str(exc)}})


if __name__ == "__main__":
    main()
