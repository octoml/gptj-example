import argparse
import csv
import logging
import typing

import gptj
import benchmark
import server

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "gptj.onnx"
DEFAULT_MODEL_NAME = "decoder_model.onnx"
DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_DEVICE = "cuda"
DEFAULT_PROMPT = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)
DEFAULT_SERVER_PORT = 8080


def benchmark_model(
    model: gptj.GPTJModel,
    prompt: str,
    sequence_lengths: typing.List[int],
    batch_sizes: typing.List[int],
    result_path: str,
):
    BENCHMARK_WARMUPS = 3
    BENCHMARK_RUNS = 10

    def get_generate_fn(sequence, batch):
        return lambda: model.generate(prompt, sequence, batch)

    with open(result_path, "w") as csv_file:
        fieldnames = ["runtime", "sequence", "batch_size", "latency_avg_ms", "qps"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for sequence_length, batch_size in (
            (s, b) for s in sequence_lengths for b in batch_sizes
        ):
            csv_result = {
                "runtime": model.runtime,
                "sequence": sequence_length,
                "batch_size": batch_size,
            }
            generate_fn = get_generate_fn(sequence_length, batch_size)
            try:
                benchmark_result = benchmark.benchmark_fn(
                    generate_fn, num_warmups=BENCHMARK_WARMUPS, num_runs=BENCHMARK_RUNS
                )
                csv_result["latency_avg_ms"] = benchmark_result.latency_avg_ms
                csv_result["qps"] = benchmark_result.qps
                logging.info(
                    f"Benchmark Scenario (Seq: {sequence_length}, Batch: {batch_size}) - Latency {benchmark_result.latency_avg_ms:.3f} ms, QPS: {benchmark_result.qps:.3f}"
                )

            except Exception as e:
                logging.warn(
                    f"Benchmark Scenario (Seq: {sequence_length}, Batch: {batch_size}) - Failed: {e}"
                )
                csv_result["latency_avg_ms"] = "NA"
                csv_result["qps"] = "NA"

            csv_writer.writerow(csv_result)
            csv_file.flush()


# Entry point
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%H:%M:%S",
        format="%(asctime)s %(levelname)-8s : %(message)s",
    )

    parser = argparse.ArgumentParser(description="GPT-J ONNX utilities.")
    subparsers = parser.add_subparsers()

    # Export the model
    def _export(args):
        gptj.export_to(args.onnx_model_path)

    export_parser = subparsers.add_parser(
        "export", help="Export the GPT-J model to onnx."
    )
    export_parser.add_argument(
        "--onnx-model-path", help="Destination path", default=DEFAULT_MODEL_PATH
    )
    export_parser.set_defaults(func=_export)

    # Convert to FP16
    def _quantize(args):
        gptj.convert_to_fp16(
            args.onnx_model_path, args.onnx_model_path, args.model_name
        )

    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize the Optimum model from FP32 to FP16"
    )
    quantize_parser.add_argument(
        "--onnx-model-path",
        help="Path to source ONNX model - e.g. /gptj.onnx/model.onnx",
        default=DEFAULT_MODEL_PATH,
    )
    quantize_parser.add_argument(
        "--model-name",
        help="Filename of the ONNX model - e.g. decoder_model.onnx",
        default=DEFAULT_MODEL_NAME,
    )
    quantize_parser.set_defaults(func=_quantize)

    # Generate text - run the model
    def _generate(args):
        model = gptj.GPTJModel.load(args.onnx_model_path, args.device)
        result = model.generate(args.prompt, args.sequence_length)
        print(result)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate text using the model"
    )
    generate_parser.add_argument(
        "--onnx-model-path",
        help="Path to exported ONNX model. Transformers PyTorch model will be used if unspecified.",
    )
    generate_parser.add_argument("--device", help="CPU or CUDA", default=DEFAULT_DEVICE)
    generate_parser.add_argument(
        "--prompt", help="Prompt to use for generating text", default=DEFAULT_PROMPT
    )
    generate_parser.add_argument(
        "--sequence-length",
        help="Length of text to generate",
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    generate_parser.set_defaults(func=_generate)

    # Benchmark the model
    def _benchmark(args):
        model = gptj.GPTJModel.load(args.onnx_model_path, args.device)
        benchmark_model(
            model,
            DEFAULT_PROMPT,
            args.sequence_lengths,
            args.batch_sizes,
            args.result_path,
        )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark the PyTorch or ONNX version of the model"
    )
    benchmark_parser.add_argument(
        "--onnx-model-path",
        help="Path to exported ONNX model. Transformers PyTorch model will be used if unspecified.",
    )
    benchmark_parser.add_argument(
        "--device", help="CPU or CUDA", default=DEFAULT_DEVICE
    )
    benchmark_parser.add_argument(
        "--sequence-lengths",
        help="Sequence length",
        nargs="*",
        type=int,
        default=[DEFAULT_SEQUENCE_LENGTH],
    )
    benchmark_parser.add_argument(
        "--batch-sizes", help="Batch sizes", nargs="*", type=int, default=[1]
    )
    benchmark_parser.add_argument(
        "--result-path", help="Path to target results csv", default="results.csv"
    )
    benchmark_parser.set_defaults(func=_benchmark)

    # Serve an API endpoint
    def _serve(args):
        model = gptj.GPTJModel.load(args.onnx_model_path, args.device)
        server.launch(args.port, model)

    serve_parser = subparsers.add_parser("serve", help="Host an endpoint")
    serve_parser.add_argument(
        "--onnx-model-path",
        help="Path to exported ONNX model. Transformers PyTorch model will be used if unspecified.",
    )
    serve_parser.add_argument("--device", help="CPU or CUDA", default=DEFAULT_DEVICE)
    serve_parser.add_argument("--port", help="Server port", default=DEFAULT_SERVER_PORT)
    serve_parser.set_defaults(func=_serve)

    # Parse and execute commands
    generate_args = "generate --onnx-model-path gptj.onnx --prompt".split()
    generate_args.append('"once upon a time, there was a little monkey,"')
    benchmark_args = "benchmark --onnx-model-path gptj.onnx --sequence-lengths 64 --batch-sizes 1 2 4".split()
    serve_args = "serve --onnx-model-path gptj.onnx".split()

    args = parser.parse_args(serve_args)
    args.func(args)
