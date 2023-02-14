This walkthrough guides the user to create an FP16 version of the [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
which can be executed via ONNX Runtime on an A10G GPU with 24GB RAM

The default FP32 version of the model requires 48GB of CPU RAM to load the model.
The FP16 version should fit on a 16GB GPU.


# Create the Python environment
sudo apt update && sudo apt install python3.8-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --no-deps -r requirements.txt

# Use the Optimun CLI to export the model to ONNX on a CPU
export ORT_CUDA_UNAVAILABLE=1
optimum-cli export onnx --model EleutherAI/gpt-j-6B --task causal-lm-with-past --device cpu --framework pt gptj.onnx/
optimum-cli export onnx --model EleutherAI/gpt-j-6B --task causal-lm --monolith --device cpu --framework pt gptj.onnx/

# Use ONNX utilities to convert the model to FP16


# Now run the model

# Here is a simple FastAPI server for the model




pip install git+https://github.com/huggingface/optimum.git@4f2ef17f1f638aab35de7cb3e186d773ef6f6ab2