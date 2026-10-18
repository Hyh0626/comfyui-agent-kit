"""Workflow and local LoRA helpers for the comfyui-local MCP server."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", r"E:\Program\ComfyUI"))
PET_PROJECT_ROOT = Path(os.environ.get("PET_PROJECT_ROOT", r"E:\Program\桌宠"))
KOHYA_ROOT = os.environ.get("KOHYA_SS_ROOT", r"E:\Program\kohya_ss")
WORKFLOW_DIR = COMFYUI_HOME / "user" / "default" / "workflows"
TRAINING_DIR = PET_PROJECT_ROOT / "training"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def tools() -> list[dict[str, Any]]:
    return [
        {"name": "comfyui_search_nodes", "description": "按名称/分类搜索本机 ComfyUI 节点，为构建工作流发现可用节点。", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "required": ["query"]}},
        {"name": "comfyui_get_node_schema", "description": "获取本机指定 ComfyUI 节点的输入、输出与控件 schema。", "inputSchema": {"type": "object", "properties": {"node_class": {"type": "string"}}, "required": ["node_class"]}},
        {"name": "comfyui_build_workflow", "description": "辅助搭建并按本机节点 schema 校验工作流草案：生成 SDXL 图生图/多视角差分 API workflow，或按 ComfyUI 官方 Hunyuan3D 单图/多视角模板生成 GLB 工作流；已有 3D API workflow 也可校验。会报告缺少的模型和输入图，不会保存、排队或执行。", "inputSchema": {"type": "object", "properties": {"workflow_type": {"type": "string", "enum": ["img2img", "view_diffs", "image_to_3d"]}, "image": {"type": "string", "description": "ComfyUI input 目录中的图片文件名；图生图或单图 3D 使用"}, "view_prompts_json": {"type": "string", "description": "视角差分 JSON 对象，键为 left/right/back，值为该视角提示词；view_diffs 使用"}, "views_json": {"type": "string", "description": "可选多视角 JSON 对象，键为 front/left/back/right，值为 ComfyUI input 图片文件名；至少两视角时使用 Hunyuan3D MV 模型"}, "checkpoint": {"type": "string", "description": "图生图的 SDXL checkpoint 文件名"}, "model_name": {"type": "string", "description": "可选 Hunyuan3D checkpoint 文件名；默认按单图/MV 模板选择"}, "mesh_resolution": {"type": "integer", "minimum": 512, "maximum": 8192, "description": "Hunyuan3D 潜空间分辨率，官方模板默认 3072"}, "positive": {"type": "string"}, "negative": {"type": "string"}, "steps": {"type": "integer", "minimum": 1, "maximum": 150}, "cfg": {"type": "number", "minimum": 0, "maximum": 30}, "denoise": {"type": "number", "minimum": 0, "maximum": 1}, "seed": {"type": "integer", "minimum": 0}, "lora_name": {"type": "string", "description": "可选：本机 LoRA 文件名"}, "lora_strength": {"type": "number", "minimum": -10, "maximum": 10}, "filename_prefix": {"type": "string"}, "workflow_json": {"type": "string", "description": "可选：已有 3D API workflow JSON；工具会用本机 schema 校验"}}, "required": ["workflow_type"]}},
        {"name": "comfyui_validate_workflow", "description": "使用本机节点列表校验 API 格式 workflow 的节点类型、必填输入和连线引用；不提交队列。", "inputSchema": {"type": "object", "properties": {"workflow_json": {"type": "string"}}, "required": ["workflow_json"]}},
        {"name": "comfyui_save_workflow", "description": "把已确认的 UI workflow 或 API workflow JSON 保存到 ComfyUI 用户工作流目录；API 文件以 .api.json 结尾。", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "workflow_json": {"type": "string"}, "format": {"type": "string", "enum": ["ui", "api"]}}, "required": ["name", "workflow_json", "format"]}},
        {"name": "comfyui_apply_pet_art", "description": "将 ComfyUI output 图片应用到桌宠项目 assets/art/<角色id>.png；覆盖现有立绘必须显式设置 overwrite=true，旧图会备份。", "inputSchema": {"type": "object", "properties": {"filename": {"type": "string"}, "subfolder": {"type": "string"}, "type": {"type": "string", "enum": ["output", "temp"]}, "character_id": {"type": "string"}, "overwrite": {"type": "boolean"}}, "required": ["filename", "character_id"]}},
        {"name": "comfyui_lora_preflight", "description": "检查桌宠训练集图片/字幕、重复文件、SDXL 底模、kohya_ss 和 GPU 显存，输出本机 LoRA 预检报告。", "inputSchema": {"type": "object", "properties": {"dataset_dir": {"type": "string"}, "base_model_path": {"type": "string"}, "trainer_root": {"type": "string"}}}},
        {"name": "comfyui_lora_prepare", "description": "复制已审核的角色图片到训练暂存目录、按触发词生成 caption 草案，并保存 SDXL LoRA 训练计划；学习率默认 1e-4，可配置训练状态保存以便中断后续训；不会启动训练。", "inputSchema": {"type": "object", "properties": {"dataset_dir": {"type": "string"}, "character_id": {"type": "string"}, "base_model_path": {"type": "string"}, "trigger_word": {"type": "string"}, "caption_text": {"type": "string", "description": "没有逐图覆盖时使用的通用描述"}, "caption_overrides_json": {"type": "string", "description": "可选 JSON 对象，按图片文件名给出不同视角/服装/表情 caption"}, "rank": {"type": "integer", "enum": [4, 8]}, "epochs": {"type": "integer", "minimum": 1, "maximum": 100}, "resolution": {"type": "integer", "enum": [512, 640, 704, 768]}, "learning_rate": {"type": "number", "minimum": 0.000001, "maximum": 0.01, "description": "训练学习率；SDXL LoRA 默认 1e-4，覆盖 sd-scripts 的 2e-6 默认值"}, "save_training_state": {"type": "boolean", "description": "每次保存 LoRA checkpoint 时也保存优化器/调度器状态，供中断续训；会占用额外磁盘空间"}, "lr_scheduler": {"type": "string", "enum": ["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"], "description": "学习率调度器；cosine_with_restarts 为硬重启余弦（每周期内 LR 从初始值余弦降到 0 再跳回）"}, "lr_scheduler_num_cycles": {"type": "integer", "minimum": 1, "maximum": 64, "description": "cosine_with_restarts 的重启周期数；24 epochs 建议 4~8"}, "lr_warmup_steps": {"type": "integer", "minimum": 0, "description": "调度器线性预热步数；0 表示不预热"}, "repeat": {"type": "integer", "minimum": 1, "maximum": 100, "description": "每张图重复次数，决定数据集目录名 <repeat>_<trigger>；默认 10"}, "network_alpha": {"type": "integer", "minimum": 1, "maximum": 128, "description": "LoRA alpha；不填时等于 rank（例如 dim16/alpha8）"}, "train_batch_size": {"type": "integer", "minimum": 1, "maximum": 8, "description": "每卡 batch size；8GB 显存建议 1"}, "optimizer_type": {"type": "string", "enum": ["Adafactor", "AdamW", "AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit"], "description": "优化器；Adafactor 省显存，AdamW8bit 质量更好（需 bitsandbytes）"}, "learning_rate_te": {"type": "number", "minimum": 0, "maximum": 0.01, "description": "Text Encoder 学习率；>0 时训练 TE（自动关闭 cache_text_encoder_outputs 与 unet_only）；传 0 或省略表示不训练 TE（走缓存路径，TE 留在 CPU，省显存）"}}, "required": ["character_id", "trigger_word", "caption_text"]}},
        {"name": "comfyui_lora_start", "description": "启动已经准备好的 SDXL LoRA 训练计划。仅在用户看过数据/caption/参数并明确确认后调用 confirmed=true；需要已安装 kohya_ss。可选 resume_state_path 从保存的完整训练状态续训。", "inputSchema": {"type": "object", "properties": {"plan_path": {"type": "string"}, "confirmed": {"type": "boolean"}, "trainer_root": {"type": "string"}, "resume_state_path": {"type": "string", "description": "可选：sd-scripts accelerator.save_state 目录；只有启用 save_training_state 并存在该目录时才能续训"}}, "required": ["plan_path", "confirmed"]}},
        {"name": "comfyui_lora_status", "description": "查看指定 LoRA 训练计划的运行状态和最近日志。", "inputSchema": {"type": "object", "properties": {"plan_path": {"type": "string"}}, "required": ["plan_path"]}},
    ]


def _json_arg(args: dict[str, Any], key: str) -> Any:
    value = json.loads(args[key])
    if not isinstance(value, dict):
        raise ValueError(f"{key} 必须是 JSON 对象")
    return value


def _within_project(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = PET_PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"数据目录必须位于桌宠项目目录内：{root}")
    if not resolved.is_dir():
        raise ValueError(f"目录不存在：{resolved}")
    return resolved


def _image_files(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name.lower())


def _trainer(root_value: str | None) -> tuple[Path, Path | None, Path | None]:
    root = Path(root_value or KOHYA_ROOT).expanduser()
    scripts = root / "sd-scripts"
    script = scripts / "sdxl_train_network.py"
    if not script.is_file():
        scripts = root
        script = root / "sdxl_train_network.py"
    python = next((p for p in (root / "venv" / "Scripts" / "python.exe", root / ".venv" / "Scripts" / "python.exe") if p.is_file()), None)
    return root.resolve(), script if script.is_file() else None, python


def _runtime_check(python: Path) -> dict[str, Any]:
    """检查训练 venv 是否具备必需的运行时模块。

    用 importlib.util.find_spec 而不是真的 import：find_spec 不加载模块，
    因此不会因为 import torch 花掉十几秒而拖慢预检。
    """
    code = (
        "import importlib.util as u\n"
        "missing = [m for m in ('torch', 'accelerate') if u.find_spec(m) is None]\n"
        "print('OK' if not missing else 'MISSING:' + ','.join(missing))\n"
    )
    try:
        result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": f"无法执行训练 venv 的 Python（{type(exc).__name__}）"}
    stdout_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    last = stdout_lines[-1] if stdout_lines else ""
    if last == "OK":
        return {"ok": True, "reason": ""}
    if last.startswith("MISSING:"):
        return {"ok": False, "reason": "训练 venv 缺少模块 " + last.split(":", 1)[1]}
    stderr_lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
    detail = stderr_lines[-1] if stderr_lines else f"exit={result.returncode}"
    return {"ok": False, "reason": "训练 venv 自检未通过：" + detail}


def _pid_alive(pid: int) -> bool:
    """判断 PID 是否仍在运行。

    不能用 os.kill(pid, 0) 做「存在性检查」：
    在 Windows 上 CPython 的 os.kill 会走 OpenProcess + TerminateProcess，
    sig=0 等价于「用退出码 0 结束该进程」—— 对正在运行的训练就是把它杀掉；
    而 PID 不存在时又会抛 WinError 87 而不是 ProcessLookupError。
    因此这里在 Windows 上改用 OpenProcess + GetExitCodeProcess 只做查询。
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            # 查不到退出码时保守地认为仍存活，避免误删 pid 记录
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _models() -> list[str]:
    found: list[str] = []
    for folder in (COMFYUI_HOME / "models" / "checkpoints", COMFYUI_HOME / "models" / "diffusion_models"):
        if folder.is_dir():
            found.extend(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".safetensors", ".ckpt"})
    return found


def _gpu_info() -> list[str] | str:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=False)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()] or result.stderr.strip() or "未检测到 NVIDIA GPU"
    except (OSError, subprocess.TimeoutExpired):
        return "无法通过 nvidia-smi 读取 GPU 状态"


def _caption_text(image: Path) -> str:
    """读取图片对应的 caption 文本。

    文件不存在、读取失败、或内容仅有空白，都视为"没有 caption"。
    """
    path = image.with_suffix(".txt")
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _dataset_report(folder: Path) -> dict[str, Any]:
    files = _image_files(folder)
    captions = {p.name: _caption_text(p) for p in files}
    missing_captions = [p.name for p in files if not captions[p.name]]
    hashes: dict[str, list[str]] = {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(path.name)
    duplicates = [names for names in hashes.values() if len(names) > 1]
    unique_captions = {text for text in captions.values() if text}
    warnings: list[str] = []
    if len(files) < 15:
        warnings.append("图片数量较少；先检查角色覆盖、画质和角度，避免直接长时间训练。")
    if duplicates:
        warnings.append("存在内容完全相同的图片。")
    if len(files) > 1 and len(unique_captions) == 1:
        warnings.append("所有 caption 完全相同；差异图应逐张描述（用 caption_overrides_json），单一通用 caption 会削弱触发词与画面特征的绑定。")
    return {
        "dataset_dir": str(folder),
        "image_count": len(files),
        "images": [
            {
                "name": p.name,
                "bytes": p.stat().st_size,
                "caption_exists": p.with_suffix(".txt").is_file(),
                "caption_chars": len(captions[p.name]),
            }
            for p in files
        ],
        "missing_captions": missing_captions,
        "exact_duplicates": duplicates,
        "unique_caption_count": len(unique_captions),
        "total_bytes": sum(p.stat().st_size for p in files),
        "warnings": warnings,
    }


def _staged_report(folder: Path) -> dict[str, Any]:
    """统计训练集图片，兼容 kohya 的 `<repeat>_<concept>` 子目录结构。

    sd-scripts 的 --train_data_dir 指向父目录，实际图片放在
    形如 `10_artoriaxl/` 的子目录里（数字前缀 = 每张图重复次数）。
    因此不能只做非递归扫描，否则会误判为「没有图片」。
    """
    if _image_files(folder):
        return _dataset_report(folder)
    merged: dict[str, Any] = {"image_count": 0, "missing_captions": [], "exact_duplicates": [], "subdirs": []}
    for sub in sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        report = _dataset_report(sub)
        if not report["image_count"]:
            continue
        merged["subdirs"].append({"name": sub.name, "image_count": report["image_count"]})
        merged["image_count"] += report["image_count"]
        merged["missing_captions"] += [f"{sub.name}/{name}" for name in report["missing_captions"]]
        merged["exact_duplicates"] += [[f"{sub.name}/{name}" for name in group] for group in report["exact_duplicates"]]
    return merged


def _preflight(dataset_dir: str | None, base_model_path: str | None, trainer_root: str | None, request: Callable[..., Any]) -> dict[str, Any]:
    folder = _within_project(dataset_dir or (PET_PROJECT_ROOT / "assets" / "refs"))
    report = _dataset_report(folder)
    root, script, python = _trainer(trainer_root)
    models = _models()
    selected_model = Path(base_model_path).expanduser().resolve() if base_model_path else None
    if selected_model and not selected_model.is_file():
        model_state = f"指定底模不存在：{selected_model}"
    elif selected_model:
        model_state = str(selected_model)
    elif models:
        model_state = "需要用户从以下候选中选择一个 SDXL 底模"
    else:
        model_state = "未发现底模；ComfyUI checkpoints/diffusion_models 目录当前没有权重文件"
    free_bytes = shutil.disk_usage(TRAINING_DIR.parent if TRAINING_DIR.parent.exists() else PET_PROJECT_ROOT).free
    try:
        comfy_queue = request("GET", "/queue")
    except Exception as exc:
        comfy_queue = {"available": False, "reason": str(exc)}

    # ---- 就绪判定 ----
    # 必须逐项通过；并把每一项未通过的具体原因（含具体图片名）报出来，
    # 避免只给一个 ready_to_train=false 让使用者无从下手。
    runtime = _runtime_check(python) if python else {"ok": False, "reason": "未找到 kohya_ss 的 Windows venv Python"}
    blockers: list[str] = []
    if not report["image_count"]:
        blockers.append("训练集目录里没有图片。")
    if report["missing_captions"]:
        names = report["missing_captions"]
        shown = "、".join(names[:8]) + ("…" if len(names) > 8 else "")
        blockers.append(f"有 {len(names)} 张图片缺少非空 caption：{shown}")
    if not selected_model:
        blockers.append("尚未指定底模；请传入 base_model_path，或从 candidate_models 中选定一个。")
    elif not selected_model.is_file():
        blockers.append(f"指定底模不存在：{selected_model}")
    if not script:
        blockers.append(f"未找到 sd-scripts 的 sdxl_train_network.py（{root}）。")
    if not python:
        blockers.append(f"未找到 kohya_ss 的 Windows venv Python（{root}\\venv\\Scripts\\python.exe）。")
    elif not runtime["ok"]:
        blockers.append(runtime["reason"])
    if report["exact_duplicates"]:
        blockers.append("存在内容完全相同的图片：" + "；".join("、".join(group) for group in report["exact_duplicates"]))

    return {
        "recommendation": "SDXL 动漫 LoRA + kohya_ss/sd-scripts；RTX 5060 Laptop 的 8GB 显存采用 U-Net-only、rank 4、batch 1、gradient checkpointing 和缓存。训练期间应退出 ComfyUI，避免争用显存。",
        "dataset": report,
        "gpu": _gpu_info(),
        "base_model": model_state,
        "candidate_models": models,
        "trainer": {
            "root": str(root),
            "installed": script is not None and python is not None,
            "script": str(script) if script else None,
            "python": str(python) if python else None,
            "runtime_ok": runtime["ok"],
            "runtime_reason": runtime["reason"],
        },
        "free_disk_bytes": free_bytes,
        "comfyui_queue": comfy_queue,
        "ready_to_train": not blockers,
        "ready_blockers": blockers,
    }


def _validate(workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    node_ids = {str(node_id) for node_id in workflow}
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            errors.append(f"节点 {node_id} 不是对象")
            continue
        class_type = node.get("class_type")
        spec = object_info.get(class_type) if isinstance(class_type, str) else None
        if not spec:
            errors.append(f"节点 {node_id}: 本机找不到节点类型 {class_type!r}")
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            errors.append(f"节点 {node_id}: inputs 必须是对象")
            continue
        required = spec.get("input", {}).get("required", {})
        for key in required:
            if key not in inputs:
                errors.append(f"节点 {node_id} ({class_type}): 缺少必填输入 {key}")
        for key, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in node_ids and isinstance(value[1], int):
                source_id = str(value[0])
                source = workflow.get(source_id, {})
                source_spec = object_info.get(source.get("class_type"), {}) if isinstance(source, dict) else {}
                output_types = source_spec.get("output", [])
                if value[1] < 0 or (output_types and value[1] >= len(output_types)):
                    errors.append(f"节点 {node_id}.{key}: 来源 {source_id} 的输出索引 {value[1]} 越界")
        declared_inputs = set(required) | set(spec.get("input", {}).get("optional", {}))
        extras = sorted(set(inputs) - declared_inputs)
        if extras:
            warnings.append(f"节点 {node_id} ({class_type}): 本机 schema 未声明输入 {', '.join(extras)}")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "node_count": len(workflow), "note": "此校验不会执行 ComfyUI prompt；节点自定义逻辑和模型文件仍需运行时验证。"}


def call_tool(name: str, args: dict[str, Any], request: Callable[..., Any]) -> Any | None:
    if name == "comfyui_search_nodes":
        query = args["query"].casefold()
        limit = max(1, min(100, int(args.get("limit", 25))))
        info = request("GET", "/object_info")
        matches = []
        for node_class, spec in info.items():
            display = str(spec.get("display_name", ""))
            category = str(spec.get(" category", spec.get("category", "")))
            if query in node_class.casefold() or query in display.casefold() or query in category.casefold():
                matches.append({"node_class": node_class, "display_name": display, "category": category, "description": spec.get("description", "")})
                if len(matches) >= limit:
                    break
        return {"query": args["query"], "count": len(matches), "nodes": matches}
    if name == "comfyui_build_workflow":
        workflow_type = args["workflow_type"]
        info = request("GET", "/object_info")
        if workflow_type == "view_diffs":
            required = ["image", "checkpoint", "positive"]
            missing = [key for key in required if not str(args.get(key, "")).strip()]
            if missing:
                raise ValueError("视角差分工作流缺少参数：" + ", ".join(missing))
            image_name = Path(args["image"]).name
            checkpoint_name = Path(args["checkpoint"]).name
            if not (COMFYUI_HOME / "input" / image_name).is_file():
                raise ValueError(f"ComfyUI input 目录中找不到参考图：{image_name}；先用 comfyui_upload_image 上传")
            if not any(Path(path).name == checkpoint_name and "checkpoints" in Path(path).parts for path in _models()):
                raise ValueError(f"本机 checkpoints 中找不到底模：{checkpoint_name}")
            prompts = json.loads(args.get("view_prompts_json", "{}"))
            if not isinstance(prompts, dict):
                raise ValueError("view_prompts_json 必须是视角到提示词的 JSON 对象")
            if not prompts:
                prompts = {
                    "left": "full body, left side view, profile, consistent outfit and character design",
                    "right": "full body, right side view, profile, consistent outfit and character design",
                    "back": "full body, back view, facing away, consistent outfit and character design",
                }
            allowed_views = {"left", "right", "back"}
            if set(prompts) - allowed_views:
                raise ValueError("视角差分只接受 left、right、back 键")
            if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts.values()):
                raise ValueError("每个视角都必须有非空提示词")
            steps = int(args.get("steps", 28))
            cfg = float(args.get("cfg", 6.0))
            denoise = float(args.get("denoise", 0.65))
            base_seed = int(args.get("seed", 20261017))
            lora_name = str(args.get("lora_name", "")).strip()
            lora_strength = float(args.get("lora_strength", 0.8))
            if not 1 <= steps <= 150 or not math.isfinite(cfg) or not 0 <= cfg <= 30 or not math.isfinite(denoise) or not 0 <= denoise <= 1 or not math.isfinite(lora_strength):
                raise ValueError("steps/cfg/denoise/lora_strength 超出允许范围")
            if base_seed < 0 or base_seed + len(prompts) - 1 > 18446744073709551615:
                raise ValueError("seed 超出 ComfyUI KSampler 允许范围")
            if lora_name and not any(Path(path).name == Path(lora_name).name for path in (COMFYUI_HOME / "models" / "loras").rglob("*") if path.is_file()):
                raise ValueError(f"本机 models/loras 中找不到 LoRA：{lora_name}")
            stage_workflows = {}
            for index, (view, view_prompt) in enumerate(prompts.items()):
                model_ref = ["10", 0] if lora_name else ["1", 0]
                clip_ref = ["10", 1] if lora_name else ["1", 1]
                graph = {
                    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint_name}},
                    "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
                    "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
                    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": ", ".join(part for part in (str(args["positive"]).strip(), view_prompt.strip()) if part), "clip": clip_ref}},
                    "5": {"class_type": "CLIPTextEncode", "inputs": {"text": str(args.get("negative", "")), "clip": clip_ref}},
                    "6": {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0], "seed": base_seed + index, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}},
                    "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                    "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"views/{view}_diff"}},
                }
                if lora_name:
                    graph["10"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora_name, "strength_model": lora_strength, "strength_clip": lora_strength}}
                stage_workflows[view] = {"workflow": graph, "validation": _validate(graph, info), "seed": base_seed + index, "output_prefix": f"views/{view}_diff"}
            return {
                "workflow_type": workflow_type,
                "stage": "generate_view_differences",
                "input_image": image_name,
                "view_prompts": prompts,
                "view_workflows": stage_workflows,
                "parameters": {"checkpoint": checkpoint_name, "lora_name": lora_name or None, "lora_strength": lora_strength if lora_name else None, "steps": steps, "cfg": cfg, "denoise": denoise},
                "handoff": "分别审阅输出的左/右/背面图；确认身份、服装与轮廓可接受后，将选定图片通过 comfyui_upload_image 放入 ComfyUI input，再用 workflow_type=image_to_3d 和 views_json 构建 Hunyuan3D MV 图。",
                "limitation": "SDXL img2img 视角提示只生成近似差分，单张图不包含真实背面信息，侧/背面可能产生身份或服装漂移；输出必须人工筛选后再进入 3D。工具不会自动排队或生成图片。",
            }
        if workflow_type == "image_to_3d":
            if args.get("workflow_json"):
                workflow = json.loads(args["workflow_json"])
                if not isinstance(workflow, dict):
                    raise ValueError("workflow_json 必须是 ComfyUI API workflow 对象")
                validation = _validate(workflow, info)
                return {"workflow_type": workflow_type, "workflow": workflow, "validation": validation, "next_step": "先确认 validation.valid=true 且 workflow 所需模型、输入图均存在，再展示并征求运行确认。"}
            views = json.loads(args.get("views_json", "{}"))
            if not isinstance(views, dict):
                raise ValueError("views_json 必须是视角到 ComfyUI input 图片文件名的 JSON 对象")
            allowed_views = {"front", "left", "back", "right"}
            if set(views) - allowed_views:
                raise ValueError("views_json 只接受 front、left、back、right 键")
            for view, filename in list(views.items()):
                if not isinstance(filename, str) or not filename.strip():
                    raise ValueError(f"视角 {view} 的图片文件名不能为空")
                views[view] = Path(filename).name
            if len(views) == 1 and "front" not in views:
                raise ValueError("单图模式请使用 image 参数或 views_json.front")
            if not views and args.get("image"):
                views = {"front": Path(args["image"]).name}
            if not views:
                raise ValueError("图生 3D 需要 image 或 views_json 中至少一张输入图")
            multiview = len(views) > 1
            default_model = "hunyuan3d-dit-v2-mv_fp16.safetensors" if multiview else "hunyuan3d-dit-v2_fp16.safetensors"
            model_name = Path(args.get("model_name") or default_model).name
            model_path = COMFYUI_HOME / "models" / "checkpoints" / model_name
            missing_images = [name for name in views.values() if not (COMFYUI_HOME / "input" / name).is_file()]
            workflow: dict[str, Any] = {
                "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": model_name}},
                "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
                "3": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": int(args.get("mesh_resolution", 3072)), "batch_size": 1}},
                "4": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["3", 0], "seed": int(args.get("seed", 0)), "steps": int(args.get("steps", 20)), "cfg": float(args.get("cfg", 8.0 if not multiview else 7.5)), "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
                "5": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["4", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": 256}},
                "6": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["5", 0], "algorithm": "surface net", "threshold": 0.6}},
                "7": {"class_type": "SaveGLB", "inputs": {"mesh": ["6", 0], "filename_prefix": str(args.get("filename_prefix", "mesh/ComfyUI"))}},
            }
            if multiview:
                for i, (view, filename) in enumerate(views.items(), start=10):
                    load_id, encode_id = str(i), str(i + 10)
                    workflow[load_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
                    workflow[encode_id] = {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": [load_id, 0], "crop": "none"}}
                    workflow["8"] = workflow.get("8", {"class_type": "Hunyuan3Dv2ConditioningMultiView", "inputs": {}})
                    workflow["8"]["inputs"][view] = [encode_id, 0]
            else:
                view, filename = next(iter(views.items()))
                workflow["9"] = {"class_type": "LoadImage", "inputs": {"image": filename}}
                workflow["10"] = {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["9", 0], "crop": "none"}}
                workflow["8"] = {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["10", 0]}}
            validation = _validate(workflow, info)
            seed = int(args.get("seed", 0))
            steps = int(args.get("steps", 20))
            cfg = float(args.get("cfg", 8.0 if not multiview else 7.5))
            if not 0 <= seed <= 18446744073709551615 or not 1 <= steps <= 150 or not math.isfinite(cfg) or not 0 <= cfg <= 30:
                raise ValueError("seed、steps 或 cfg 超出允许范围")
            if not 512 <= int(args.get("mesh_resolution", 3072)) <= 8192:
                raise ValueError("mesh_resolution 必须在 512 到 8192 之间")
            return {
                "workflow_type": workflow_type,
                "workflow": workflow,
                "validation": validation,
                "template_source": "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_hunyuan3d_image_to_model.json" if not multiview else "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/3d_hunyuan3d_multiview_to_model.json",
                "mode": "hunyuan3d_v2_multiview" if multiview else "hunyuan3d_v2_single_view",
                "model_checkpoint": model_name,
                "missing_dependencies": ([f"模型文件缺失：{model_path}"] if not model_path.is_file() else []) + [f"输入图未上传：{name}" for name in missing_images],
                "note": "此官方 Hunyuan3D 2.0 模板输出 GLB 几何网格；原生模板不负责纹理材质或骨骼绑定，因此结果不等于 VRM 桌宠模型。8GB 显存运行时可能紧张。工作流仅为草案，不会保存或排队。",
                "next_step": "检查 validation.valid 与 missing_dependencies；把输入图上传并安装列出的 checkpoint 后，先保存 API workflow，用户确认后再排队运行。",
            }
        if workflow_type != "img2img":
            raise ValueError(f"不支持的 workflow_type：{workflow_type}")
        required = ["image", "checkpoint", "positive", "negative"]
        missing = [key for key in required if not str(args.get(key, "")).strip()]
        if missing:
            raise ValueError("图生图缺少必填参数：" + ", ".join(missing))
        image_name = Path(args["image"]).name
        checkpoint_name = Path(args["checkpoint"]).name
        checkpoint_files = _models()
        if not any(Path(path).name == checkpoint_name and "checkpoints" in Path(path).parts for path in checkpoint_files):
            raise ValueError(f"本机 checkpoints 中找不到底模：{checkpoint_name}")
        if not (COMFYUI_HOME / "input" / image_name).is_file():
            raise ValueError(f"ComfyUI input 目录中找不到图片：{image_name}；先用 comfyui_upload_image 上传图片")
        steps = int(args.get("steps", 28))
        cfg = float(args.get("cfg", 6.0))
        denoise = float(args.get("denoise", 0.45))
        seed = int(args.get("seed", 0))
        lora_name = str(args.get("lora_name", "")).strip()
        lora_strength = float(args.get("lora_strength", 0.8))
        if not 1 <= steps <= 150 or not math.isfinite(cfg) or not 0 <= cfg <= 30 or not math.isfinite(denoise) or not 0 <= denoise <= 1 or not 0 <= seed <= 18446744073709551615 or not math.isfinite(lora_strength):
            raise ValueError("steps/cfg/denoise/seed 超出允许范围")
        if lora_name and not any(Path(path).name == Path(lora_name).name for path in (COMFYUI_HOME / "models" / "loras").rglob("*") if path.is_file()):
            raise ValueError(f"本机 models/loras 中找不到 LoRA：{lora_name}")
        model_ref = ["10", 0] if lora_name else ["1", 0]
        clip_ref = ["10", 1] if lora_name else ["1", 1]
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint_name}},
            "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "3": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": args["positive"], "clip": clip_ref}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": args["negative"], "clip": clip_ref}},
            "6": {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}},
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": str(args.get("filename_prefix", "ComfyUI_img2img"))}},
        }
        if lora_name:
            workflow["10"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": lora_name, "strength_model": lora_strength, "strength_clip": lora_strength}}
        validation = _validate(workflow, info)
        return {"workflow_type": workflow_type, "workflow": workflow, "validation": validation, "parameters": {"image": image_name, "checkpoint": checkpoint_name, "steps": steps, "cfg": cfg, "denoise": denoise, "seed": seed, "lora_name": lora_name or None, "lora_strength": lora_strength if lora_name else None}, "next_step": "此为 API workflow 草案，不会自动运行或保存；先查看图与参数，再调用 comfyui_save_workflow(format=api) 或在用户确认后调用 comfyui_queue_workflow。"}
    if name == "comfyui_get_node_schema":
        node_class = urllib.parse.quote(args["node_class"], safe="")
        return request("GET", "/object_info/" + node_class)
    if name == "comfyui_validate_workflow":
        workflow = _json_arg(args, "workflow_json")
        return _validate(workflow, request("GET", "/object_info"))
    if name == "comfyui_save_workflow":
        workflow = _json_arg(args, "workflow_json")
        fmt = args["format"]
        if fmt == "ui" and not isinstance(workflow.get("nodes"), list):
            raise ValueError("UI workflow 需要包含 nodes 数组；API prompt map 请设置 format=api")
        if fmt == "api" and any(not isinstance(v, dict) or "class_type" not in v for v in workflow.values()):
            raise ValueError("API workflow 必须是节点 ID 到含 class_type 的节点对象映射")
        stem = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", Path(args["name"]).stem).strip("_.")
        if not stem:
            raise ValueError("工作流名称不能为空")
        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        suffix = ".api.json" if fmt == "api" else ".json"
        target = WORKFLOW_DIR / (stem + suffix)
        if target.exists():
            raise FileExistsError(f"工作流已存在，未覆盖：{target}")
        target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"saved_to": str(target), "format": fmt, "note": "UI workflow 可在 ComfyUI 打开；API workflow 是用于 MCP 队列的 API 图，不能当作画布格式直接打开。"}
    if name == "comfyui_apply_pet_art":
        character_id = args["character_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", character_id):
            raise ValueError("character_id 仅允许字母、数字、下划线和连字符")
        filename = args["filename"]
        if Path(filename).suffix.lower() != ".png":
            raise ValueError("桌宠立绘目标使用 PNG；请先从 ComfyUI output 选择 PNG 文件")
        query = urllib.parse.urlencode({"filename": filename, "subfolder": args.get("subfolder", ""), "type": args.get("type", "output")})
        req = urllib.request.Request(os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/") + "/view?" + query)
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()
        target = PET_PROJECT_ROOT / "assets" / "art" / (character_id + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not args.get("overwrite", False):
                raise FileExistsError(f"立绘已存在；展示预览后若确认替换，再设置 overwrite=true：{target}")
            backup = target.with_name(target.stem + ".before-comfyui.png")
            if not backup.exists():
                shutil.copy2(target, backup)
        target.write_bytes(content)
        return {"saved_to": str(target), "bytes": len(content), "backup": str(target.with_name(target.stem + ".before-comfyui.png")) if target.with_name(target.stem + ".before-comfyui.png").exists() else None}
    if name == "comfyui_lora_preflight":
        return _preflight(args.get("dataset_dir"), args.get("base_model_path"), args.get("trainer_root"), request)
    if name == "comfyui_lora_prepare":
        character_id = args["character_id"]
        trigger = args["trigger_word"].strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", character_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", trigger):
            raise ValueError("character_id 和 trigger_word 仅允许字母、数字、下划线和连字符")
        source = _within_project(args.get("dataset_dir") or (PET_PROJECT_ROOT / "assets" / "refs"))
        images = _image_files(source)
        if not images:
            raise ValueError(f"目录内没有可训练图片：{source}")
        selected = Path(args["base_model_path"]).expanduser().resolve() if args.get("base_model_path") else None
        if selected and not selected.is_file():
            raise ValueError(f"底模文件不存在：{selected}")
        model_candidates = _models()
        if not selected and model_candidates:
            selected = Path(model_candidates[0])
        rank = int(args.get("rank", 4))
        epochs = int(args.get("epochs", 8))
        resolution = int(args.get("resolution", 768))
        learning_rate = float(args.get("learning_rate", 1e-4))
        if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-2:
            raise ValueError("learning_rate 必须是 1e-6 到 1e-2 之间的有限数值；SDXL LoRA 默认推荐 1e-4")
        save_training_state = args.get("save_training_state", True)
        if not isinstance(save_training_state, bool):
            raise ValueError("save_training_state 必须是布尔值")
        lr_scheduler = str(args.get("lr_scheduler", "constant")).strip() or "constant"
        if lr_scheduler not in ("constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"):
            raise ValueError("lr_scheduler 只支持 constant/linear/cosine/cosine_with_restarts/polynomial/constant_with_warmup")
        lr_scheduler_num_cycles = int(args.get("lr_scheduler_num_cycles", 1))
        if not 1 <= lr_scheduler_num_cycles <= 64:
            raise ValueError("lr_scheduler_num_cycles 必须在 1 到 64 之间")
        lr_warmup_steps = int(args.get("lr_warmup_steps", 0))
        if lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps 不能为负数")
        repeat = int(args.get("repeat", 10))
        if not 1 <= repeat <= 100:
            raise ValueError("repeat 必须在 1 到 100 之间")
        network_alpha = int(args.get("network_alpha", rank))
        if not 1 <= network_alpha <= 128:
            raise ValueError("network_alpha 必须在 1 到 128 之间")
        train_batch_size = int(args.get("train_batch_size", 1))
        if not 1 <= train_batch_size <= 8:
            raise ValueError("train_batch_size 必须在 1 到 8 之间")
        optimizer_type = str(args.get("optimizer_type", "Adafactor")).strip() or "Adafactor"
        if optimizer_type not in ("Adafactor", "AdamW", "AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit"):
            raise ValueError("optimizer_type 只支持 Adafactor/AdamW/AdamW8bit/PagedAdamW8bit/Lion/Lion8bit")
        # 填了 learning_rate_te 就等于「同时训练 Text Encoder」：必须去掉 unet_only 与 TE 输出缓存
        raw_lr_te = args.get("learning_rate_te")
        if raw_lr_te is None or raw_lr_te == "":
            train_text_encoder = False
            learning_rate_te = None
        else:
            learning_rate_te = float(raw_lr_te)
            if learning_rate_te == 0:
                # TE 学习率 0 == 不训练 TE。走缓存路径把 TE 留在 CPU，否则它会被搬上 GPU 拖垮 8GB 显存。
                train_text_encoder = False
                learning_rate_te = None
            elif not math.isfinite(learning_rate_te) or not 1e-6 <= learning_rate_te <= 1e-2:
                raise ValueError("learning_rate_te 必须是 0（不训练 TE）或 1e-6 到 1e-2 之间的有限数值")
            else:
                train_text_encoder = True
        caption_overrides = json.loads(args.get("caption_overrides_json", "{}"))
        if not isinstance(caption_overrides, dict):
            raise ValueError("caption_overrides_json 必须是文件名到 caption 的 JSON 对象")
        base_caption = args.get("caption_text", "").strip()

        def image_caption(image: Path) -> str:
            # caption 优先级：逐图覆盖 > 源目录同名 .txt（已人工审阅）> 通用兜底 caption_text
            if image.name in caption_overrides:
                description = str(caption_overrides[image.name]).strip()
            else:
                description = _caption_text(image) or base_caption
            return ", ".join(part for part in (trigger, description) if part)
        run_dir = TRAINING_DIR / character_id
        if run_dir.exists():
            raise FileExistsError(f"训练准备目录已存在；为防止覆盖，请先选新的 character_id 或手动整理：{run_dir}")
        dataset = run_dir / "dataset" / f"{repeat}_{trigger}"
        dataset.mkdir(parents=True)
        for image in images:
            shutil.copy2(image, dataset / image.name)
            (dataset / (image.stem + ".txt")).write_text(image_caption(image) + "\n", encoding="utf-8")
        output_dir = run_dir / "output"
        output_dir.mkdir()
        plan = {
            "character_id": character_id,
            "source_dataset": str(source),
            "dataset_dir": str(run_dir / "dataset"),
            "staged_images": len(images),
            "trigger_word": trigger,
            "caption_text": base_caption,
            "caption_overrides": caption_overrides,
            "base_model_path": str(selected) if selected else None,
            "output_dir": str(output_dir),
            "output_name": character_id + "_sdxl_lora",
            "rank": rank,
            "network_alpha": network_alpha,
            "epochs": epochs,
            "repeat": repeat,
            "train_batch_size": train_batch_size,
            "optimizer_type": optimizer_type,
            "train_text_encoder": train_text_encoder,
            "learning_rate_te": learning_rate_te,
            "resolution": f"{resolution},{resolution}",
            "learning_rate": learning_rate,
            "save_training_state": save_training_state,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_num_cycles": lr_scheduler_num_cycles,
            "lr_warmup_steps": lr_warmup_steps,
            "trainer_root": KOHYA_ROOT,
            "recommendation": "SDXL U-Net-only LoRA；默认 learning_rate=1e-4（显式覆盖 sd-scripts 的 2e-6 默认值）、batch 1、gradient checkpointing、缓存 latents/text encoder outputs、Adafactor、低 rank。训练状态保存可用于中断续训，但会额外占用磁盘。8GB 显存是紧配置，先确认底模、captions 和训练参数。",
            "warnings": (["当前只有少量训练图，可能过拟合；检查多角度/构图覆盖。"] if len(images) < 15 else []) + (["没有检测到底模；先安装/放入 SDXL checkpoint 并重新准备计划。"] if not selected else []) + (["已启用每个 LoRA checkpoint 的训练状态保存，便于中断续训；会额外占用磁盘空间。"] if save_training_state else []),
            "confirmed_for_training": False,
        }
        plan_path = run_dir / "training-plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"plan_path": str(plan_path), "plan": plan, "captions_preview": [{"image": p.name, "caption": image_caption(p)} for p in images[:8]], "next_step": "逐张审阅 captions 和底模；只有准备开训时才调用 comfyui_lora_start，并设置 confirmed=true。"}
    if name == "comfyui_lora_start":
        if args.get("confirmed") is not True:
            raise ValueError("训练不会启动：先展示并确认数据、caption、底模、参数和显存占用，再设置 confirmed=true")
        plan_path = Path(args["plan_path"]).expanduser().resolve()
        if TRAINING_DIR.resolve() not in plan_path.parents:
            raise ValueError("plan_path 必须位于桌宠项目 training 目录")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        learning_rate = float(plan.get("learning_rate", 1e-4))
        if not math.isfinite(learning_rate) or not 1e-6 <= learning_rate <= 1e-2:
            raise ValueError("训练计划 learning_rate 必须是 1e-6 到 1e-2 之间的有限数值")
        rank = int(plan.get("rank", 4))
        network_alpha = int(plan.get("network_alpha", rank) or rank)
        if not 1 <= network_alpha <= 128:
            raise ValueError("训练计划 network_alpha 必须在 1 到 128 之间")
        train_batch_size = int(plan.get("train_batch_size", 1) or 1)
        if not 1 <= train_batch_size <= 8:
            raise ValueError("训练计划 train_batch_size 必须在 1 到 8 之间")
        grad_accum = int(plan.get("gradient_accumulation_steps", 1) or 1)
        if not 1 <= grad_accum <= 64:
            raise ValueError("训练计划 gradient_accumulation_steps 必须在 1 到 64 之间")
        save_every_n_epochs = int(plan.get("save_every_n_epochs", 1) or 1)
        if not 1 <= save_every_n_epochs <= 100:
            raise ValueError("训练计划 save_every_n_epochs 必须在 1 到 100 之间")
        optimizer_type = str(plan.get("optimizer_type", "Adafactor") or "Adafactor")
        if optimizer_type not in ("Adafactor", "AdamW", "AdamW8bit", "PagedAdamW8bit", "Lion", "Lion8bit"):
            raise ValueError(f"训练计划 optimizer_type 非法：{optimizer_type}")
        train_text_encoder = bool(plan.get("train_text_encoder", False))
        learning_rate_te = plan.get("learning_rate_te")
        if train_text_encoder:
            if learning_rate_te is None:
                raise ValueError("训练计划开启了 train_text_encoder 但缺少 learning_rate_te")
            learning_rate_te = float(learning_rate_te)
            if not math.isfinite(learning_rate_te) or not 1e-6 <= learning_rate_te <= 1e-2:
                raise ValueError("训练计划 learning_rate_te 必须是 1e-6 到 1e-2 之间的有限数值")
        save_training_state = plan.get("save_training_state", True)
        if not isinstance(save_training_state, bool):
            raise ValueError("训练计划 save_training_state 必须是布尔值")
        lr_scheduler = str(plan.get("lr_scheduler", "constant") or "constant")
        if lr_scheduler not in ("constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"):
            raise ValueError(f"训练计划 lr_scheduler 非法：{lr_scheduler}")
        lr_scheduler_num_cycles = int(plan.get("lr_scheduler_num_cycles", 1) or 1)
        if not 1 <= lr_scheduler_num_cycles <= 64:
            raise ValueError("训练计划 lr_scheduler_num_cycles 必须在 1 到 64 之间")
        lr_warmup_steps = int(plan.get("lr_warmup_steps", 0) or 0)
        if lr_warmup_steps < 0:
            raise ValueError("训练计划 lr_warmup_steps 不能为负数")
        # SDXL + fp16 的 VAE 有 NaN 风险；默认让 VAE 走 fp32
        no_half_vae = bool(plan.get("no_half_vae", True))
        raw_min_snr = plan.get("min_snr_gamma")
        if raw_min_snr is None or raw_min_snr == "":
            min_snr_gamma = None
        else:
            min_snr_gamma = float(raw_min_snr)
            if not math.isfinite(min_snr_gamma) or not 0 < min_snr_gamma <= 100:
                raise ValueError("训练计划 min_snr_gamma 必须是 (0, 100] 之间的有限数值")
        # 分层 rank / 分层学习率（SDXL 的 block 总数固定为 23：1+9+3+9+1）
        network_args = []
        raw_bd = str(plan.get("block_dims") or "").strip()
        raw_ba = str(plan.get("block_alphas") or "").strip()
        if raw_bd or raw_ba:
            if not raw_bd or not raw_ba:
                raise ValueError("必须同时提供 block_dims 与 block_alphas：sd-scripts 会断言两者长度一致")
            bd_list = [x for x in raw_bd.split(",") if x.strip()]
            ba_list = [x for x in raw_ba.split(",") if x.strip()]
            if len(bd_list) != len(ba_list):
                raise ValueError(f"block_dims 有 {len(bd_list)} 个值，block_alphas 有 {len(ba_list)} 个，必须相等")
            if len(bd_list) != 23:
                raise ValueError(f"SDXL 的 block_dims 必须是 23 个值（1+9+3+9+1），当前 {len(bd_list)} 个")
            network_args.append("block_dims=" + raw_bd)
            network_args.append("block_alphas=" + raw_ba)
        for _key in ("down_lr_weight", "mid_lr_weight", "up_lr_weight"):
            _val = str(plan.get(_key) or "").strip()
            if _val:
                network_args.append(_key + "=" + _val)
        model = Path(plan["base_model_path"]) if plan.get("base_model_path") else None
        if not model or not model.is_file():
            raise ValueError("训练计划没有可用 SDXL 底模；请先放入 checkpoint 并重新准备计划")
        trainer_root, script, python = _trainer(args.get("trainer_root") or plan.get("trainer_root"))
        if not script or not python:
            raise RuntimeError(f"kohya_ss/sd-scripts 未安装或缺少 Windows venv：{trainer_root}。预检和计划已可用，安装后重试。")

        # 不依赖预检标志，启动前再次核对数据集、caption 与运行环境，
        # 防止"预检通过后 caption 被删/清空"或"计划目录被替换"导致带病开训。
        staged = Path(plan.get("dataset_dir") or "").expanduser()
        if not staged.is_dir():
            raise ValueError(f"训练计划里的 dataset_dir 不存在：{staged}")
        staged_report = _staged_report(staged)
        if not staged_report["image_count"]:
            raise ValueError(f"训练集目录里没有图片：{staged}")
        if staged_report["missing_captions"]:
            names = staged_report["missing_captions"]
            shown = "、".join(names[:8]) + ("…" if len(names) > 8 else "")
            raise ValueError(f"训练不会启动：有 {len(names)} 张图片缺少非空 caption：{shown}。补齐后重新预检。")
        if staged_report["exact_duplicates"]:
            raise ValueError("训练不会启动：数据集存在内容完全相同的图片：" + "；".join("、".join(group) for group in staged_report["exact_duplicates"]))
        runtime = _runtime_check(python)
        if not runtime["ok"]:
            raise RuntimeError("训练不会启动：" + runtime["reason"])
        run_dir = plan_path.parent
        pid_file = run_dir / "training.pid"
        if pid_file.exists():
            try:
                old_pid = int(json.loads(pid_file.read_text(encoding="utf-8"))["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                old_pid = 0
            if _pid_alive(old_pid):
                raise RuntimeError(f"此角色已有训练进程运行，PID={old_pid}")
            # 进程已退出：清理陈旧的 pid 记录后继续
            pid_file.unlink(missing_ok=True)
        log_path = run_dir / "training.log"
        script_args = [
            str(python), "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1", str(script),
            "--pretrained_model_name_or_path", str(model), "--train_data_dir", plan["dataset_dir"],
            "--output_dir", plan["output_dir"], "--output_name", plan["output_name"],
            "--resolution", plan["resolution"], "--enable_bucket", "--train_batch_size", str(train_batch_size),
            "--max_train_epochs", str(plan["epochs"]), "--network_module", "networks.lora",
            "--learning_rate", format(learning_rate, ".12g"),
            "--network_dim", str(rank), "--network_alpha", str(network_alpha),
            "--gradient_checkpointing", "--cache_latents",
            "--gradient_accumulation_steps", str(grad_accum),
            "--optimizer_type", optimizer_type,
            "--mixed_precision", "fp16", "--save_model_as", "safetensors", "--save_every_n_epochs", str(save_every_n_epochs),
            "--caption_extension", ".txt",
        ]
        if optimizer_type == "Adafactor":
            script_args.extend(["--optimizer_args", "scale_parameter=False", "relative_step=False", "warmup_init=False"])
        if train_text_encoder:
            # 训练 Text Encoder：必须去掉 unet_only 与 TE 输出缓存（sd-scripts 有硬断言），
            # 并用 --unet_lr / --text_encoder_lr 分别指定两端学习率。
            script_args.extend([
                "--unet_lr", format(learning_rate, ".12g"),
                "--text_encoder_lr", format(float(learning_rate_te), ".12g"),
            ])
        else:
            script_args.append("--network_train_unet_only")
            script_args.append("--cache_text_encoder_outputs")
        # 学习率调度器：不显式指定时维持 sd-scripts 默认的 constant，行为与扩展前完全一致。
        if lr_scheduler != "constant":
            script_args.extend(["--lr_scheduler", lr_scheduler])
            if lr_scheduler == "cosine_with_restarts":
                script_args.extend(["--lr_scheduler_num_cycles", str(lr_scheduler_num_cycles)])
        if lr_warmup_steps > 0:
            script_args.extend(["--lr_warmup_steps", str(lr_warmup_steps)])
        # SDXL 的 VAE 在 fp16 下可能产生 NaN（概率性），默认关掉半精度 VAE
        if no_half_vae:
            script_args.append("--no_half_vae")
        # Min-SNR 损失加权：小数据集 LoRA 的稳定化手段
        if min_snr_gamma is not None:
            script_args.extend(["--min_snr_gamma", format(min_snr_gamma, ".12g")])
        if save_training_state:
            script_args.append("--save_state")
        # 分层 rank / 分层学习率：必须放在最后，避免 --network_args 吞掉后续参数
        if network_args:
            script_args.append("--network_args")
            script_args.extend(network_args)
        resume_state_path = args.get("resume_state_path")
        if resume_state_path:
            resume_state = Path(resume_state_path).expanduser().resolve()
            if not save_training_state:
                raise ValueError("该计划未启用 save_training_state，无法从优化器状态续训")
            if not resume_state.is_dir():
                raise ValueError(f"续训状态目录不存在：{resume_state}")
            output_dir = Path(plan["output_dir"]).expanduser().resolve()
            if output_dir not in resume_state.parents:
                raise ValueError(f"续训状态目录必须位于本训练计划的 output_dir 内：{output_dir}")
            script_args.extend(["--resume", str(resume_state)])
        log = log_path.open("ab")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        proc = subprocess.Popen(script_args, cwd=str(script.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
        log.close()
        pid_file.write_text(json.dumps({"pid": proc.pid, "log_path": str(log_path), "started_at": __import__("datetime").datetime.now().astimezone().isoformat()}, indent=2), encoding="utf-8")
        return {"started": True, "pid": proc.pid, "log_path": str(log_path), "command": script_args, "learning_rate": learning_rate, "lr_scheduler": lr_scheduler, "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None, "lr_warmup_steps": lr_warmup_steps, "save_training_state": save_training_state, "resumed_from": str(resume_state) if resume_state_path else None, "note": "训练在独立进程运行；ComfyUI 与训练共用 8GB 显存，请勿同时加载大模型。"}
    if name == "comfyui_lora_status":
        plan_path = Path(args["plan_path"]).expanduser().resolve()
        if TRAINING_DIR.resolve() not in plan_path.parents:
            raise ValueError("plan_path 必须位于桌宠项目 training 目录")
        pid_file = plan_path.parent / "training.pid"
        if not pid_file.is_file():
            return {"running": False, "reason": "没有训练进程记录", "plan_path": str(plan_path)}
        record = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = int(record["pid"])
        # 注意：Windows 上 os.kill(pid, 0) 会真的 TerminateProcess 掉该进程，
        # 用它做「是否在运行」的检查会把正在跑的训练杀掉，因此改用只读查询。
        running = _pid_alive(pid)
        log_path = Path(record["log_path"])
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:] if log_path.is_file() else []
        return {"running": running, "pid": pid, "started_at": record.get("started_at"), "log_path": str(log_path), "recent_log": tail}
    return None
