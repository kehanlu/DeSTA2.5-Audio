#!/usr/bin/env python3
"""
Start vLLM OpenAI-compatible server with DeSTA2.5-Audio model.

Usage:
    python start_server.py
    python start_server.py --model /path/to/local/model --port 8000
    python start_server.py --tensor-parallel-size 2 --gpu-memory-utilization 0.9
"""

# Register DeSTA25 model with vLLM (must be before starting server)
import desta.vllm  # noqa: E402, F401

from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


def main():
    parser = FlexibleArgumentParser(description="DeSTA2.5-Audio vLLM Server")
    parser = make_arg_parser(parser)

    # Set defaults for DeSTA2.5-Audio
    parser.set_defaults(
        model="DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B",
        tokenizer="DeSTA-ntu/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_model_len=4096,
        host="0.0.0.0",
        port=8000,
        memory_utilization=0.7,
    )

    args = parser.parse_args()

    import uvloop
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
