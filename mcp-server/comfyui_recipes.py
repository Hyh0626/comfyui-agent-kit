# -*- coding: utf-8 -*-
"""ComfyUI 生成配方（可复用）

封装本机实测有效的做法：
  full_body_hiresfix()  —— 全身图 + 二段放大重绘，修复小脸上瞳孔偏移
  portrait_face()       —— 面部特写，方形画布，避免切脸

用法：
    from comfyui_recipes import full_body_hiresfix, ANIMAGINE, LORA_E12P2, NEG_BASE
    wf = full_body_hiresfix(
        positive="takeuchi_style, ...",
        negative=NEG_BASE,
        lora=LORA_E12P2, lora_strength=0.85,
        seed=20261018,
    )
    # 把 wf 传给 comfyui_queue_workflow（json.dumps(wf)）

实测依据（RTX 5060 Laptop / 8151 MiB）：
  - 832x1216 全身单次生成：脸约 100px、眼睛约 15px → 瞳孔易漂
  - 832x1216 → 放大 1248x1824 → img2img denoise 0.40：瞳孔恢复正常
  - 面部图用 1024x1024 + portrait 可避免 close-up 切脸
"""

ANIMAGINE = "animagine-xl-4.0-zero.safetensors"

# ComfyUI 里 lora_name 必须用反斜杠分隔子目录
LORA_E12 = r"takeuchi_style_v2\takeuchi_style_v2_sdxl_lora-000012.safetensors"
LORA_E12P1 = r"takeuchi_e12_plus1\e12(1).safetensors"
LORA_E12P2 = r"takeuchi_e12_plus2\e12(2).safetensors"

# 通用负向：质量 + 去重 + 年龄/气质 + 圆脸
NEG_BASE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, malformed limbs, "
    "fused limbs, twisted body, text, error, cropping, worst quality, low quality, "
    "signature, watermark, blurry, multiple views, "
    "2boys, multiple boys, 2girls, multiple girls, duplicate, twins, extra person, "
    "child, loli, immature, too young, small eyes, squinting, "
    "seductive, suggestive, sultry, flirtatious, heavy makeup, nsfw, "
    "tareme, round face, chubby, baby face, moe"
)

# 五官修正（HiresFix 第二段追加）
NEG_FACE_FIX = (
    "deformed pupils, asymmetrical eyes, cross-eyed, extra pupils, heterochromia, "
    "dead eyes, malformed face, cropped face, out of frame"
)


def _lora_nodes(g, model_ref, clip_ref, lora, strength):
    """插入 LoraLoader，返回 (model_ref, clip_ref)"""
    if not lora:
        return model_ref, clip_ref
    g["20"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "model": model_ref,
            "clip": clip_ref,
            "lora_name": lora,
            "strength_model": strength,
            "strength_clip": strength,
        },
    }
    return ["20", 0], ["20", 1]


def _ckpt(g, ckpt):
    g["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}


def full_body_hiresfix(
    positive,
    negative=NEG_BASE,
    ckpt=ANIMAGINE,
    lora=LORA_E12P2,
    lora_strength=0.85,
    width=832,
    height=1216,
    upscale=1.5,
    denoise=0.40,
    steps=28,
    cfg=5.5,
    seed=20261018,
    step2_steps=None,
    prefix="hires/full_body",
    extra_face_neg=NEG_FACE_FIX,
):
    """全身图两段式：先生成，再放大并低 denoise 重绘（修复小脸瞳孔、统一画风）。

    返回可直接 json.dumps 后交给 comfyui_queue_workflow 的 API workflow。
    """
    w2 = int(width * upscale) // 8 * 8
    h2 = int(height * upscale) // 8 * 8
    neg2 = negative + (", " + extra_face_neg if extra_face_neg else "")
    g = {}
    _ckpt(g, ckpt)
    g["2"] = {"class_type": "EmptyLatentImage",
              "inputs": {"width": int(width), "height": int(height), "batch_size": 1}}
    mref, cref = _lora_nodes(g, ["1", 0], ["1", 1], lora, lora_strength)
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": cref}}
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": cref}}
    g["6"] = {"class_type": "KSampler", "inputs": {
        "model": mref, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["2", 0],
        "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
        "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["8"] = {"class_type": "ImageScale", "inputs": {
        "image": ["7", 0], "upscale_method": "lanczos",
        "width": w2, "height": h2, "crop": "disabled"}}
    g["9"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}}
    g["5b"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg2, "clip": cref}}
    g["10"] = {"class_type": "KSampler", "inputs": {
        "model": mref, "positive": ["4", 0], "negative": ["5b", 0], "latent_image": ["9", 0],
        "seed": int(seed), "steps": int(step2_steps or steps), "cfg": float(cfg),
        "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": float(denoise)}}
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
    g["12"] = {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}}
    return g


def portrait_face(
    positive,
    negative=NEG_BASE,
    ckpt=ANIMAGINE,
    lora=LORA_E12P2,
    lora_strength=0.85,
    size=1024,
    steps=28,
    cfg=5.5,
    seed=20261018,
    prefix="hires/face",
):
    """面部/表情图：方形画布 + portrait，避免 close-up 在竖幅下切掉半张脸。

    提示词里的构图词请用 `portrait, upper body`，不要用 `close-up`；
    表情 tag 最多 2 个（如 `angry, frown`），堆多了会夸张变形。
    """
    g = {}
    _ckpt(g, ckpt)
    g["2"] = {"class_type": "EmptyLatentImage",
              "inputs": {"width": int(size), "height": int(size), "batch_size": 1}}
    mref, cref = _lora_nodes(g, ["1", 0], ["1", 1], lora, lora_strength)
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": cref}}
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative + ", " + NEG_FACE_FIX, "clip": cref}}
    g["6"] = {"class_type": "KSampler", "inputs": {
        "model": mref, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["2", 0],
        "seed": int(seed), "steps": int(steps), "cfg": float(cfg),
        "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}}
    g["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}}
    return g


# ---- 提示词骨架（按经验固定顺序，便于改写）----
# [1] 触发词 + 质量词
# [2] 角色身份 + 外观 + 成年感(young woman) + 脸型(large eyes, (tsurime:0.6))
# [3] 服装 + 构图（full body / upper body 必须显式写，否则触发词会吸收构图）
# [4] 视角 + 背景
STYLE_HEAD = "takeuchi_style, masterpiece, high score, great score, absurdres, newest, "

SAKURA_KIMONO_FRONT = (
    STYLE_HEAD
    + "1girl, solo, matou sakura, fate (series), purple hair, purple eyes, "
      "hair ribbon, red ribbon, young woman, large eyes, (tsurime:0.6), detailed eyes, "
      "kimono, japanese clothes, obi, long sleeves, wide sleeves, full body, "
      "from front, facing viewer, white background, simple background"
)

__all__ = [
    "ANIMAGINE", "LORA_E12", "LORA_E12P1", "LORA_E12P2",
    "NEG_BASE", "NEG_FACE_FIX", "STYLE_HEAD", "SAKURA_KIMONO_FRONT",
    "full_body_hiresfix", "portrait_face",
]
