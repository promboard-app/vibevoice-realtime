"""promboard.vibevoice-realtime — VibeVoice-Realtime-0.5B(Microsoft) 한 화자 목소리 노드 코드.

노드 하나(vibevoice-realtime)가 글을 읽어 한 화자의 목소리 WAV(24kHz 모노)를 낸다. 목소리는 공식 프리셋 25개 중 하나를 고른다
(영어 7 + 독일어 · 프랑스어 · 이탈리아어 · 일본어 · 한국어 · 네덜란드어 · 폴란드어 · 포르투갈어 · 스페인어 실험 목소리 각 2).

공식 경로
  microsoft/VibeVoice GitHub(커밋 16fb2cb, MIT)의 vibevoice 패키지를 nodes/vibevoice 에 그대로 담았다(ASR 파일은 뺐고, 레지스트리
  검사에 걸리는 오류 글 한 줄만 바꿨다 — 그 파일 머리 주석). 이 노드는 공식 예제 demo/realtime_model_inference_from_file.py 와
  같은 순서로 부른다: 모델 · 프로세서 → set_ddpm_inference_steps → 목소리 프리셋(torch.load weights_only=True) →
  process_input_with_cached_prompt → generate(cfg_scale, do_sample=False, all_prefilled_outputs).
  공식 코드는 transformers 4.51.3 에 묶여 있어(5.x 미만) VibeVoice-1.5B 플러그인과 따로 낸다.
  목소리 프리셋 .pt 는 공식 저장소 demo/voices/streaming_model 의 파일을 PromBoard 릴리스로 다시 올려 받는다(MIT, README).
  글 토크나이저는 원본 설정(language_model_pretrained_name)이 가리키는 Qwen/Qwen2.5-0.5B 의 토크나이저 파일만 받아, 공식
  from_pretrained 와 같은 값으로 프로세서를 직접 만든다(인터넷 없이 읽으려고).
  주의: 공식 예제는 CUDA 에서 flash_attention_2 를 쓰고 SDPA 는 '조금 낮은 품질일 수 있다'고 적는다. flash-attn 은 세 OS 휠이 없어
  이 노드는 SDPA 를 쓴다(공식 예제의 대체 경로).
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import copy
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 받은 파일만 쓴다(실행 환경은 인터넷이 막혀 있다)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # 담아 온 vibevoice 패키지를 그 이름 그대로 불러온다(공식 코드가 절대 경로로 import 한다)

MODEL_FILE = "microsoft/VibeVoice-Realtime-0.5B/config.json"
TOKENIZER_FILE = "Qwen/Qwen2.5-0.5B/tokenizer.json"
SAMPLE_RATE = 24000
TOKENS_PER_SEC = 7.5  # 소리 토큰 하나 = 1/7.5초(공식 카드 7.5Hz)
MAX_TEXT_CHARS = 5000  # 한 번에 읽을 글 상한(약 5분 분량) — 문맥 8k 토큰 · 앱이 노드 한 번에 주는 시간 안에 끝나게

_STATE = {}


def _load(ctx):
    device = ctx.device
    if _STATE.get("device") == device:
        return _STATE["model"], _STATE["processor"]
    import torch
    from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor

    ctx.progress(0.02, "load-model")  # 올리는 동안 파이썬이 오래 멈춘다 — 바로 앞에 알린다
    start = time.monotonic()
    _STATE.clear()
    folder = os.path.dirname(ctx.model_path(MODEL_FILE))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # 공식 예제: CUDA bf16, MPS · CPU float32
    try:
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            folder,
            torch_dtype=dtype,
            device_map=device if device in ("cuda", "cpu") else None,
            attn_implementation="sdpa",
        )
        if device not in ("cuda", "cpu"):
            model.to(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("그래픽카드 메모리가 모자라 모델을 올리지 못했습니다(약 3GB 필요). 다른 AI 작업 · 프로그램을 닫고 다시 해 주세요.") from None
    model.eval()
    # 공식 from_pretrained(preprocessor_config.json) 과 같은 값: 압축 3200 · dB 정규화, 소리 24kHz · -25dBFS
    processor = VibeVoiceStreamingProcessor(
        tokenizer=VibeVoiceTextTokenizerFast.from_pretrained(os.path.dirname(ctx.model_path(TOKENIZER_FILE))),
        audio_processor=VibeVoiceTokenizerProcessor(sampling_rate=SAMPLE_RATE, normalize_audio=True, target_dB_FS=-25, eps=1e-6),
        speech_tok_compress_ratio=3200,
        db_normalize=True,
    )
    _STATE.update(device=device, model=model, processor=processor)
    ctx.log(f"VibeVoice-Realtime-0.5B loaded on {device} in {time.monotonic() - start:.1f}s")
    return model, processor


def _voice(ctx, name):
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    key = ("voice", name, ctx.device)
    if key not in _STATE:
        path = ctx.model_path(f"voices/{name}.pt")
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            _STATE[key] = torch.load(path, map_location=ctx.device, weights_only=True)
    return _STATE[key]


def vibevoice_realtime(ctx, inputs, settings):
    import numpy as np
    import soundfile as sf
    import torch

    text = (inputs.get("text") or "").strip()
    if not text:
        raise ValueError("읽을 글을 이어 주세요.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"글이 너무 깁니다({len(text)}자). 한 번에 {MAX_TEXT_CHARS}자까지 읽습니다 — 나눠서 이어 주세요.")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')  # 공식 예제와 같은 따옴표 정리

    model, processor = _load(ctx)
    model.set_ddpm_inference_steps(num_steps=int(settings["steps"]))
    prefilled = _voice(ctx, settings["voice"])
    ctx.check_cancel()
    enc = processor.process_input_with_cached_prompt(
        text=text, cached_prompt=prefilled, padding=True, return_tensors="pt", return_attention_mask=True
    )
    for k, v in enc.items():
        if torch.is_tensor(v):
            enc[k] = v.to(ctx.device)

    # 글 길이로 어림한 소리 토큰 수(영어 약 15자/초)를 진행률 분모로 쓴다 — 넘으면 0.95 에서 멈춘다.
    expected = max(30, int(len(text) / 15 * TOKENS_PER_SEC))
    step = {"n": 0}

    def stop_check():
        ctx.check_cancel()
        step["n"] += 1
        if step["n"] % 10 == 0:
            ctx.progress(0.1 + 0.85 * min(1.0, step["n"] / expected), "generate")
        return False

    if ctx.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(settings["seed"]))
    start = time.monotonic()
    try:
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=None,
                cfg_scale=float(settings["cfg"]),
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                all_prefilled_outputs=copy.deepcopy(prefilled),
                stop_check_fn=stop_check,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("그래픽카드 메모리가 모자랍니다. 다른 AI 작업 · 프로그램을 닫고 다시 하거나, 글을 나눠 짧게 읽혀 주세요.") from None
    took = time.monotonic() - start
    speech = out.speech_outputs[0] if out.speech_outputs else None
    if speech is None or speech.numel() == 0:
        raise RuntimeError("목소리가 만들어지지 않았습니다. 글을 바꿔 다시 해 주세요.")
    wav = speech.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
    path = ctx.out_path("vibevoice-realtime.wav")
    sf.write(path, wav, SAMPLE_RATE, subtype="PCM_16")
    peak = torch.cuda.max_memory_reserved() / 2**20 if ctx.device == "cuda" else 0
    ctx.log(f"voice {settings['voice']}, {len(text)} chars → {wav.size / SAMPLE_RATE:.1f}s audio in {took:.1f}s, peak VRAM reserved {peak:.0f} MiB")
    ctx.progress(1.0, "done")
    return {"audio": path}


NODES = {"vibevoice-realtime": vibevoice_realtime}
