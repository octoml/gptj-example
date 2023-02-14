# gptj-example


sudo apt update && sudo apt install python3.8-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --no-deps -r requirements.txt


optimum-cli export onnx --model EleutherAI/gpt-j-6B --task causal-lm --monolith --framework pt gptjoptimum/
optimum-cli export onnx --model EleutherAI/gpt-j-6B --task causal-lm --framework pt gptj.onnx/
optimum-cli export onnx --model EleutherAI/gpt-j-6B --task causal-lm-with-past --framework pt gptjoptimum/


pip install git+https://github.com/huggingface/optimum.git@4f2ef17f1f638aab35de7cb3e186d773ef6f6ab2





For the “demo” artifacts - this is what I have in progress
Instructions to setup the environment - install appropriate versions of ORT, Hugging Face Transformers and Optimum
A script to pull the source model from HF and convert to ORT using Optimum and FP16 using Onnx tools
A command to benchmark - produce a CSV (do we care about the tokenize and decode steps here?)
A command to run a FastAPI server
A Docker container which does all of the above and launches a server
I am not at the moment - thinking about steps to deploy this container to AWS. I assume there are enough instructions on the internet to do that?