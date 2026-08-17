"""Launch the inference server."""

import asyncio
import dataclasses
import json
import logging
import os
import sys
import warnings

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

logger = logging.getLogger(__name__)

suppress_noisy_warnings()
logging.basicConfig(level=logging.INFO)


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.smg_grpc_mode or server_args.grpc_mode:
            logger.info(
                "run_server: encoder_only, gRPC path "
                f"(smg_grpc_mode={server_args.smg_grpc_mode}, "
                f"grpc_mode={server_args.grpc_mode}) -> serve_grpc_encoder"
            )
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            logger.info(
                "run_server: encoder_only, HTTP path "
                "(smg_grpc_mode=False, grpc_mode=False) -> encode_server.launch_server"
            )
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        logger.info(
            "run_server: legacy SMG gRPC path "
            f"(smg_grpc_mode={server_args.smg_grpc_mode}) -> serve_grpc"
        )
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        logger.info(
            f"run_server: Ray path (use_ray={server_args.use_ray}) "
            "-> ray.http_server.launch_server"
        )
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        logger.info("run_server: default HTTP path -> http_server.launch_server")
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from sglang.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    logger.info(
        "server_args=%s",
        json.dumps(
            dataclasses.asdict(server_args), indent=2, sort_keys=True, default=str
        ),
    )

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
