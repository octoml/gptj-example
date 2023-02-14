# ONNX Runtime scripts for GPT-J

This repository contains scripts to export, convert, benchmark and host the [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj) model using the [ONNX Runtime](https://onnxruntime.ai/).

It demonstrates how to 
- Export the model using [Optimum](https://huggingface.co/docs/optimum/index), 
- Convert to FP16 via [onnx-converters](https://github.com/microsoft/onnxconverter-common)
- Benchmark latency and QPS for a provided set of batch sizes and sequence lengths
- Host the model using [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/)


## Setup
Prepare a Python environment via the following.
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --no-deps -r requirements.txt
```

## Export the model
```python src/main.py export --onnx-model-path gptj.onnx```

## Convert to FP16
```python src/main.py convert --onnx-model-path gptj.onnx```

## Test
Test the model via
```
python src/main.py generate --onnx-model-path gptj.onnx --prompt "Once upon a time there was a Large Language Model, one day" --sequence-length 128
```

## Benchmark
The model can be benchmarked with a set of sequence lengths and batch sizes
```
python src/main.py benchmark --onnx-model-path gptj.onnx --sequence-length 64 96 128 --batch-sizes 1 2 4 --result-csv results.csv
```
It will prodiuce a CSV file with results - e.g.
```
runtime,sequence,batch_size,latency_avg_ms,qps
onnxruntime,64,1,3996.8519095999995,1.3762222827373463
onnxruntime,64,2,4363.9566114,1.2603736437585085
onnxruntime,64,4,6775.1043152,0.8115817048319952
onnxruntime,96,1,11467.769732800001,0.4782945437220458
onnxruntime,96,2,14451.319966400002,0.3805963612133471
onnxruntime,96,4,24834.637286499998,0.22174991892275242
onnxruntime,128,1,19026.3660427,0.2892998730774599
onnxruntime,128,2,26112.432011700002,0.21065564754932553
onnxruntime,128,4,46067.58433620001,0.11938218952255279
```

## Host an endpoint
The following command will start a server with an endpoint to invoke the model.
`python src/main.py serve --onnx-model-path gptj.onnx --port 8080`

It can be queried via cURL
`curl -G http://localhost:8080/generate --data-urlencode prompt="Once upon a time there was a Large Language Model, one day" --data-urlencode length=100`
