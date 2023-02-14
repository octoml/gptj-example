import argparse
import logging

import gptj

logger = logging.getLogger(__name__)


DEFAULT_PROMPT = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)
DEFAULT_SEQUENCE_LENGTH = 128


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)-8s : %(message)s",
    )

    parser = argparse.ArgumentParser(description="GPT-J utilities")
    subparsers = parser.add_subparsers()

    # Download the model
    def _download(args):
        gptj.download_to(args.model_path)

    download_parser = subparsers.add_parser("download", help="Acquire the GPT-J model")
    download_parser.add_argument(
        "--model-path", help="Destination path", default="gptj.onnx"
    )
    download_parser.set_defaults(func=_download)

    # Convert to FP16
    def _convert(args):
        gptj.convert_to_fp16(args.model_path, args.dest_path, args.model_name)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert the Optimum model from FP32 to FP16"
    )
    convert_parser.add_argument(
        "--model-path", help="Path to source ONNX model - e.g. /gptj.onnx/model.onnx"
    )
    convert_parser.add_argument(
        "--model-name",
        help="Filename of the ONNX model - e.g. /gptj.onnx/model.onnx",
        default="decoder_model.onnx",
    )
    convert_parser.add_argument("--dest-path", required=True)
    convert_parser.set_defaults(func=_convert)

    # Generate text - run the model
    def _generate(args):
        model = gptj.GPTJModel.load(args.model_path, args.device)
        result = model.generate(args.prompt, args.sequence_length)
        print(result)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text using the model"
    )
    generate_parser.add_argument("--model-path", help="Path to ONNX model")
    generate_parser.add_argument("--device", help="CPU or CUDA", default="CUDA")
    generate_parser.add_argument(
        "--prompt", help="Prompt for text generation", default=DEFAULT_PROMPT
    )
    generate_parser.add_argument(
        "--sequence-length", help="Sequence length", default=64
    )
    generate_parser.set_defaults(func=_generate)

    # Benchmark the model
    def _benchmark(args):
        gptj.benchmark(args.model_path, args.device, args.result_path)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the PyTorch or ONNX version of the model"
    )
    benchmark_parser.add_argument("--model-path", help="Path to ONNX model")
    benchmark_parser.add_argument("--device", help="CPU or CUDA", default="CUDA")
    benchmark_parser.add_argument(
        "--sequence_lengths", help="Sequence length", default="64, 128"
    )
    benchmark_parser.add_argument("--batch_size", help="Batch sizes", default="1, 2, 4")
    benchmark_parser.add_argument(
        "--result-path", help="Path to target results csv", default="results.csv"
    )
    benchmark_parser.set_defaults(func=_benchmark)

    # Serve an API endpoint
    def _serve(args):
        model = gptj.load(args.model_path, args.device)
        # server.host(args.port, model)
        pass

    serve_parser = subparsers.add_parser("serve", help="Host an endpoint")
    serve_parser.add_argument("--model-path", help="Path to ONNX model")
    serve_parser.add_argument("--device", help="cpu or cuda", default="cuda")
    serve_parser.add_argument("--port", help="Server port", default=8080)
    serve_parser.set_defaults(func=_serve)

    # Query a running API endpoint
    def _query(args):
        # client.query(args.url, args.text, args.length)
        pass

    query_parser = subparsers.add_parser("query", help="Host an endpoint")
    query_parser.add_argument(
        "--url", help="Server URL", default="http://localhost:8080"
    )
    query_parser.add_argument(
        "--text", help="Input text", default="http://localhost:8080"
    )
    query_parser.add_argument(
        "--length", help="Length to generate", default="http://localhost:8080"
    )
    query_parser.set_defaults(func=_query)

    # Parse and execute commands
    download_args = "download --model-path gptj.onnx".split()
    convert_args = "convert --model-path gptj.onnx --dest-path gptj.onnx".split()
    generate_args = "generate --model-path gptj.onnx".split()

    args = parser.parse_args(generate_args)
    args.func(args)
