import time
import typing

import numpy as np
import transformers
import torch.cuda
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

"""
python -m transformers.onnx --model=EleutherAI/gpt-j-6B --framework pt --feature=causal-lm-with-past gptjonnx
optimum-cli export onnx --model EleutherAI/gpt-j-6B --for-ort --task causal-lm-with-past gptjoptimum/

model_inputs['input_ids'].shape
torch.Size([1, 46])
model_inputs['attention_mask'].shape
torch.Size([1, 46])

"""

USE_ONNX = True
ONNX_PATH = "./gptj.fp16.onnx/"
# ONNX_PATH = "./gptjoptimum.fp16/"
# TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider

ONNX_EP = "CUDAExecutionProvider"
ONNX_EP_OPTIONS = None

# ONNX_EP = "TensorrtExecutionProvider"
# ONNX_EP_OPTIONS = {
#     "trt_max_workspace_size": 4294967296,  # 2147483648,
#     "trt_fp16_enable": True,
# }

# GENERATE_LENGTH = 64
GENERATE_LENGTH = 128


DEFAULT_WARMUP_COUNT = 4
DEFAULT_RUN_COUNT = 20
# DEFAULT_WARMUP_COUNT = 1
# DEFAULT_RUN_COUNT = 1


def benchmark_fn(
    fn: typing.Callable,
    num_warmups: int = DEFAULT_WARMUP_COUNT,
    num_runs: int = DEFAULT_RUN_COUNT,
) -> typing.List[float]:
    for _ in range(num_warmups):
        fn()
    durations_ms = []
    for i in range(num_runs):
        start_ns = time.perf_counter_ns()
        fn()
        elapsed_ns = time.perf_counter_ns() - start_ns
        elapsed_ms = float(elapsed_ns / 1e6)
        durations_ms.append(elapsed_ms)
        print(f"Iteration {i}, {elapsed_ms:0.3f}")
    return durations_ms


transformers.set_seed(1234)


start_ns = time.perf_counter_ns()
if not USE_ONNX:
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    # model = model.to("cuda")
else:
    model = ORTModelForCausalLM.from_pretrained(
        ONNX_PATH,
        # provider="TensorrtExecutionProvider",  # CPUExecutionProvider, CUDAExecutionProvider
        provider=ONNX_EP,
        provider_options=ONNX_EP_OPTIONS,
        from_transformers=False,
        force_download=False,
        local_files_only=True,
        use_cache=False,  # Skips load of "decoder_with_past_session" - What is the purpose of this?
    )
elapsed_ns = time.perf_counter_ns() - start_ns
print(f"Model Loaded - {elapsed_ns/1e6} ms")

start_ns = time.perf_counter_ns()
if not USE_ONNX:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
else:
    tokenizer = AutoTokenizer.from_pretrained(ONNX_PATH)
elapsed_ns = time.perf_counter_ns() - start_ns
print(f"Tokenizer Loaded - {elapsed_ns/1e6} ms")


input_prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley,llsin the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_tokenized = None
output_tokenized = None
output_text = None


def tokenize_text_pt():
    global input_tokenized
    # input_tokenized = tokenizer(input_prompt, return_tensors="np")
    input_tokenized = tokenizer(input_prompt, return_tensors="pt")
    input_tokenized = input_tokenized.to("cuda")


def generate_text_pt():
    global input_tokenized
    global output_tokenized
    # with torch.cuda.amp.autocast():
    output_tokenized = model.generate(
        input_tokenized.input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=GENERATE_LENGTH,
    )


def decode_tokens_pt():
    global output_tokenized
    global output_text
    output_text = tokenizer.batch_decode(output_tokenized)


tokenize_text_pt()
generate_text_pt()
decode_tokens_pt()
print(output_text)


tokenize_times = benchmark_fn(tokenize_text_pt)
tokenize_mean = np.mean(tokenize_times)
tokenize_std = np.std(tokenize_times)
print(f"Tokenize - {tokenize_mean:0.3f} ms, {tokenize_std:0.3f}")

generate_times = benchmark_fn(generate_text_pt)
generate_mean = np.mean(generate_times)
generate_std = np.std(generate_times)
print(f"Generate - {generate_mean:0.3f} ms, {generate_std:0.3f}")

decode_times = benchmark_fn(decode_tokens_pt)
decode_mean = np.mean(decode_times)
decode_std = np.std(decode_times)
print(f"Decode - {decode_mean:0.3f} ms, {decode_std:0.3f}")
