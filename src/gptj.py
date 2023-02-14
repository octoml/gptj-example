import os
import logging
import dataclasses
import typing

from optimum.onnxruntime import ORTModel, ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GPTJModel:
    tokenizer: AutoTokenizer
    model: typing.Union[ORTModelForCausalLM, AutoModelForCausalLM]
    device: str

    @classmethod
    def load(
        cls,
        model_path: typing.Optional[str] = None,
        device: typing.Optional[str] = None,
    ):
        if model_path:
            if not os.path.isdir(model_path):
                raise ValueError(f"{model_path} is not an existing directory.")
            return cls.from_exported_onnx_path(model_path=model_path, device=device)
        else:
            return cls.from_transformers(device=device)

    @classmethod
    def from_transformers(cls, device: typing.Optional[str] = None):
        transformers_model_id = "EleutherAI/gpt-j-6B"
        model = AutoModelForCausalLM.from_pretrained(transformers_model_id)
        if device:
            device = device.strip().lower()
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_id)
        logger.info(f"Loaded Transfomer model {transformers_model_id} on {device}.")
        return cls(model=model, tokenizer=tokenizer, device=device)

    @classmethod
    def from_exported_onnx_path(
        cls, model_path: str, device: typing.Optional[str] = None
    ):
        device = device.strip().lower() if device else device
        execution_provider = GPTJModel._get_onnx_ep(device)
        model = ORTModelForCausalLM.from_pretrained(
            model_path,
            provider=execution_provider,
            from_transformers=False,
            force_download=False,
            local_files_only=True,
            use_cache=False,  # Skips load of "decoder_with_past_session"..
        )
        logger.info(f"Loaded ONNX model at {model_path} using {execution_provider}.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @staticmethod
    def _get_onnx_ep(device: typing.Optional[str] = None) -> str:
        if device:
            if device.find("cuda") != -1:
                return "CUDAExecutionProvider"
            elif device.find("trt") != -1 or device.find("tensorrt") != -1:
                return "TensorrtExecutionProvider"
        return "CPUExecutionProvider"

    @property
    def runtime(self) -> str:
        if isinstance(self.model, ORTModel):
            return "onnxruntime"
        return "pytorch"

    def generate(self, prompt: str, sequence_length: int, batch_size: int = 1) -> str:
        if batch_size > 1:
            prompt = [prompt] * batch_size
        input_tokenized = self.tokenizer(prompt, return_tensors="pt")
        if self.device:
            input_tokenized = input_tokenized.to(self.device)
        output_tokenized = self.model.generate(
            input_tokenized.input_ids,
            do_sample=True,
            temperature=0.9,
            max_length=sequence_length,
        )
        output_text = self.tokenizer.batch_decode(output_tokenized)
        return output_text


def export_to(dest_dir: str):
    import subprocess
    import sys

    export_command = f"{sys.executable} -m optimum.exporters.onnx --model EleutherAI/gpt-j-6B --task causal-lm --atol 1e-04 --framework pt {dest_dir}"
    # Explicitly disable CUDA so we use CPU for the export. The GPU RAM requirements are quite high otherwise.
    export_env = os.environ.copy()
    export_env["ORT_CUDA_UNAVAILABLE"] = "1"
    subprocess.run(
        export_command,
        shell=True,
        check=True,
        env=export_env,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    logger.info(f"Exported model to {dest_dir}.")


def convert_to_fp16(
    source_folder: str, dest_folder: str, model_name="decoder_model.onnx"
):
    import onnx
    from onnxconverter_common.float16 import convert_float_to_float16_model_path

    source_model_path = os.path.join(source_folder, model_name)
    logger.info(f"Converting model at {source_model_path}.")

    fp16_model = convert_float_to_float16_model_path(source_model_path)
    logger.info(f"Converted model {source_model_path} to FP16.")

    os.makedirs(dest_folder, exist_ok=True)
    dest_model_path = os.path.join(dest_folder, model_name)
    dest_external_data_path = os.path.splitext(model_name)[0] + ".onnx_data"
    onnx.save_model(
        fp16_model,
        dest_model_path,
        save_as_external_data=True,
        location=dest_external_data_path,
    )
    logger.info(f"Saved FP16 model to {dest_model_path}.")
