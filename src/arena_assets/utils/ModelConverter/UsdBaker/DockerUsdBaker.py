from __future__ import annotations

import itertools
import subprocess
from collections.abc import Generator, Iterable
from pathlib import Path

from . import UsdBaker


def line_reader(stream: Iterable[bytes]) -> Generator[bytes, None, None]:
    buffer = bytearray()
    for chunk in stream:
        for byte in chunk:
            if byte == ord("\n"):
                yield bytes(buffer)
                buffer.clear()
            else:
                buffer.append(byte)
    return None


class DockerUsdBaker(UsdBaker):
    def __open_subprocess(self) -> subprocess.Popen:
        volumes = self._volumes.copy()
        volumes[str(self.converter_script())] = "/converter.py:ro"
        return subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "-e",
                "ACCEPT_EULA=Y",
                *itertools.chain.from_iterable(("-v", f"{host_path}:{container_path}") for host_path, container_path in volumes.items()),
                "--gpus",
                "all",
                "--entrypoint",
                "/isaac-sim/python.sh",
                self._image,
                "/converter.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def start(self):
        self._process = self.__open_subprocess()
        self.wait_for_ready()

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        image: str = "nvcr.io/nvidia/isaac-sim:5.1.0",
    ):
        super().__init__(input_dir, output_dir)
        self._image: str = image

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._volumes: dict[str, str] = {
            str(input_dir.resolve()): "/input:ro",
            str(output_dir.resolve()): "/output:rw",
        }

        self._process: subprocess.Popen | None = None

    def convert(self, input_file: str, output_file: str, retries: int = 3) -> bool:
        input_full_path = self.input_dir / input_file
        if not input_full_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_full_path}")

        output_full_path = self.output_dir / output_file
        output_full_path.parent.mkdir(parents=True, exist_ok=True)

        # Container paths
        container_input = f"/input/{input_file}"
        container_output = f"/output/{output_file}"

        # Execute conversion
        cmd = f"{container_input}:{container_output}"
        output = self.command(cmd)

        if output is None:
            if retries > 0:
                self.logger.error("Conversion failed. Process output closed unexpectedly.")
                self.logger.info(
                    "Restarting USD Baker and retrying conversion (attempts left: %d)...",
                    retries,
                )
                self.start()
                return self.convert(input_file, output_file, retries - 1)
            raise RuntimeError("Conversion failed. Process output closed unexpectedly.")

        if "error:" in output.lower():
            self.logger.error("Conversion failed: %s", output)
            return False

        self.logger.info("Conversion succeeded.")
        return True
