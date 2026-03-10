# GPU Agnostic acceleration

Source: https://deepwiki.com/search/does-this-support-rocm-or-vulc_c935dd0b-4dfd-4cff-a071-8cd39a204d06?mode=deep

# DeepWiki Q&A with Code Context for Repository: mfalfaruqi/piper-tts-server
## Q1
Does this support ROCm or Vulcan for piper to support AMD GPUs?
### Answer
## No ROCm or Vulkan Support

Based on a thorough search of the entire codebase (which consists of only `server.py` and `README.md`), there is **no support for ROCm or Vulkan** for AMD GPUs. There is zero mention of ROCm, Vulkan, HIP, or OpenCL anywhere in the project.

The only GPU acceleration currently implemented is a check for **CUDA (NVIDIA)** availability via PyTorch: piper-tts-server:24-26 

The LitServe server is launched with `accelerator="auto"`, which delegates device selection to LitServe: piper-tts-server:116-119 

This `"auto"` setting in LitServe typically defaults to CUDA/NVIDIA GPU detection and does **not** automatically enable ROCm or Vulkan.

---

## Notes

- **ROCm**: AMD's ROCm platform can sometimes work with PyTorch if you install the ROCm-specific build of PyTorch (`torch` with ROCm support). However, this is **not configured or documented** in this project, and `torch.cuda.is_available()` would still need to return `True` (which it can under ROCm with the right PyTorch build, as ROCm exposes a CUDA-compatible API).
- **Vulkan**: Piper itself uses **ONNX Runtime** for inference. ONNX Runtime does have a DirectML/Vulkan execution provider, but this project makes no use of it — models are loaded via `PiperVoice.load()` with no custom execution provider configuration.
- **Piper's ONNX Runtime**: For real AMD GPU acceleration, you would need to configure ONNX Runtime with a ROCm or Vulkan/DirectML execution provider when loading the model, which is not currently implemented here.
### Citations
**File:** server.py (L24-26)
```python
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("CUDA is available, GPU acceleration is enabled")
```
**File:** server.py (L116-119)
```python
    server = ls.LitServer(
        api,
        accelerator="auto",
    )
```
## Q2
what would be the best approach to enable AMD GPU Support. Maybe also with keeping in mind that Intel Compatibility via vulcan would be great to
### Answer
## Piper Model Loading and Inference in `piper-tts-server`

### 1. Model Discovery and Loading

At server startup, the `setup(device)` method of the `PiperTTSAPI` class scans the `models/` directory for all `*.onnx` files using `glob.glob`. For each ONNX file, it looks for a matching `*.onnx.json` config file. If both are present, it calls `PiperVoice.load(onnx_path, config_path=json_path)` and stores the resulting voice object in a dictionary (`self.models`) keyed by the model's base filename (without the `.onnx` extension). The first successfully loaded model is automatically designated as the default. piper-tts-server:17-43 

### 2. Inference (Speech Synthesis)

During each request, the `predict(inputs)` method:
1. Retrieves the appropriate `PiperVoice` object from `self.models` using the requested model name.
2. Generates a UUID-based temporary filename under `output/`.
3. Calls `voice_obj.synthesize_wav(text, wav_file)` to write the synthesized audio to that WAV file.
4. Returns the file path to `encode_response`. piper-tts-server:73-86 

### 3. Audio Encoding

`encode_response(wav_path)` loads the temporary WAV via `pydub`, converts to MP3 if needed (using `ffmpeg` under the hood), deletes the temporary file, and returns a `fastapi.Response` with the appropriate `media_type`. piper-tts-server:88-111 

---

### 4. ONNX Runtime Execution Provider Configuration — What the Codebase Actually Does

This is the critical part: **the codebase does NOT explicitly configure any ONNX Runtime execution provider at runtime**. Here is what actually happens:

#### CUDA Detection is "Dead Code"

`torch.cuda.is_available()` is called and its result is stored in `use_cuda`, but **this variable is never passed to `PiperVoice.load()` or anywhere else**. It only triggers a `print()` statement. piper-tts-server:24-26 

#### Execution Provider is Selected Entirely by Package Install

The only mechanism for switching execution providers is **which `onnxruntime` package is installed**:

- **CPU (default):** The standard `Dockerfile` installs `onnxruntime` (CPU-only) from `requirements.txt`.
- **NVIDIA CUDA GPU:** `Dockerfile.cuda` explicitly installs `onnxruntime-gpu` on top of a `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` base image.

There is no `CUDAExecutionProvider`, `ROCMExecutionProvider`, `VulkanExecutionProvider`, or any other provider string passed anywhere in the code. piper-tts-server:37-37 

#### Docker Compose GPU Toggle

Switching between CPU and GPU deployments requires manually editing `docker-compose.yml` to change the `dockerfile:` line and uncomment the NVIDIA GPU device reservation block. This is the only "configuration option" the project exposes for GPU acceleration.

---

### 5. What Is NOT Available: ROCm (AMD) and Vulkan (Intel)

The codebase has **zero support** for alternative execution providers such as:

| Provider | Purpose | Status in this codebase |
|---|---|---|
| `CUDAExecutionProvider` | NVIDIA GPU | Implicitly used via `onnxruntime-gpu` package |
| `ROCMExecutionProvider` | AMD GPU (ROCm) | ❌ Not implemented |
| `VulkanExecutionProvider` | Cross-platform GPU (Intel, AMD, etc.) | ❌ Not implemented |
| `DmlExecutionProvider` | DirectML (Windows GPU) | ❌ Not implemented |
| `TensorrtExecutionProvider` | NVIDIA TensorRT | ❌ Not implemented |

To add ROCm or Vulkan support, you would need to:
1. Install `onnxruntime-rocm` (for AMD) or a Vulkan-capable build of ONNX Runtime.
2. Check whether `PiperVoice.load()` from the upstream `piper-tts` library accepts an `use_gpu` or provider argument (this is not exposed or used in this codebase at all).
3. Pass the desired provider string (e.g., `ROCMExecutionProvider`) explicitly, which `server.py` currently never does. piper-tts-server:1-13 

---

## Summary Flow

```mermaid
graph TB
    "Startup" --> "setup(device)"
    "setup(device)" --> "glob models/*.onnx"
    "glob models/*.onnx" --> "PiperVoice.load(onnx_path, json_path)"
    "PiperVoice.load(onnx_path, json_path)" --> "self.models[base_name]"
    "HTTP POST /v1/audio/speech" --> "decode_request()"
    "decode_request()" --> "predict(inputs)"
    "predict(inputs)" --> "voice_obj.synthesize_wav(text, wav_file)"
    "voice_obj.synthesize_wav(text, wav_file)" --> "output/output_uuid.wav"
    "output/output_uuid.wav" --> "encode_response(wav_path)"
    "encode_response(wav_path)" --> "Response MP3 or WAV"

    "Which onnxruntime pkg installed?" --> "CPU: onnxruntime"
    "Which onnxruntime pkg installed?" --> "NVIDIA: onnxruntime-gpu"
    "Which onnxruntime pkg installed?" --> "AMD ROCm: NOT SUPPORTED"
    "Which onnxruntime pkg installed?" --> "Vulkan/Intel: NOT SUPPORTED"
```

## Notes

- The `use_cuda = torch.cuda.is_available()` check at line 24–26 of `server.py` is effectively **dead code** — it is computed but never used to influence model loading or inference.
- GPU acceleration is selected entirely at the **package level** (`onnxruntime` vs `onnxruntime-gpu`), not through any runtime API call.
- The `PiperVoice` class comes from the external `piper-tts` library, which is not present in this repository. Whether `PiperVoice.load()` accepts execution provider arguments depends on that library's API — but this codebase never passes any such argument.
- Adding ROCm or Vulkan support would require changes both to the installed `onnxruntime` build and to how `PiperVoice.load()` is called, neither of which is currently done.
### Citations
**File:** server.py (L1-13)
```python
import litserve as ls
from piper import PiperVoice
import os
import io
import wave
from fastapi import Response, HTTPException
from pydub import AudioSegment
from dotenv import load_dotenv
import glob
import uuid
import torch

load_dotenv()
```
**File:** server.py (L17-43)
```python
    def setup(self, device):
        """Load all available models from the ./models directory.
        Models are stored in a dict ``self.models`` keyed by the base filename
        (without the .onnx extension). The default model can still be set via
        ``DEFAULT_PIPER_MODEL`` environment variables.
        """

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("CUDA is available, GPU acceleration is enabled")

        self.models = {}
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        # Scan for *.onnx files and load corresponding config
        for onnx_path in glob.glob(os.path.join(models_dir, "*.onnx")):
            base = os.path.splitext(os.path.basename(onnx_path))[0]
            json_path = onnx_path + ".json"
            if not os.path.exists(json_path):
                continue
            try:
                self.models[base] = PiperVoice.load(onnx_path, config_path=json_path)
                print(f"Loaded model '{base}' from {onnx_path}")
            except Exception as e:
                print(f"Failed to load model {base}: {e}")

        # Store a reference to the default model name (first loaded if not set)
        self.default_model = next(iter(self.models)) if self.models else None
```
**File:** server.py (L73-86)
```python
    def predict(self, inputs):
        text = inputs["text"]
        model_name = inputs["model"]
        # Retrieve the appropriate PiperVoice instance
        if model_name not in self.models:
            raise HTTPException(
                status_code=400, detail=f"Model '{model_name}' not found."
            )
        voice_obj = self.models[model_name]
        output_path = f"output/output_{uuid.uuid4()}.wav"
        # Synthesize to a WAV buffer with proper header
        with wave.open(output_path, "wb") as wav_file:
            voice_obj.synthesize_wav(text, wav_file)
        return output_path
```
**File:** server.py (L88-111)
```python
    def encode_response(self, wav_path):
        """Encode the raw WAV bytes to the requested format.
        Uses the response format stored during decode (self.response_format).
        If ``self.response_format`` is "wav", returns raw WAV bytes.
        Otherwise converts to MP3 using pydub.
        """

        # load file
        audio = AudioSegment.from_wav(wav_path)

        # Clean up
        os.remove(wav_path)

        response_format = getattr(self, "response_format", "mp3")
        if response_format == "wav":
            return Response(content=audio.raw_data, media_type="audio/wav")
        # Convert WAV to MP3
        try:
            mp3_io = io.BytesIO()
            audio.export(mp3_io, format="mp3")
            return Response(content=mp3_io.getvalue(), media_type="audio/mpeg")
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            raise HTTPException(status_code=500, detail="Audio conversion failed")
```