import subprocess
import time
from pathlib import Path

from . import UsdBaker


class LocalUsdBaker(UsdBaker):
    """Handle USD baking locally."""

    def _open_subprocess(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "LocalUsdBaker",
                str(self.converter_script()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            executable=str(self._isaacsim_path),
        )

    def start(self):
        self._process = self._open_subprocess()
        self.logger.info(f"Started Isaac Sim subprocess with PID: {self._process.pid}. Waiting for it to be ready...")
        now = time.time()
        self.wait_for_ready()
        self.logger.info(f"Isaac Sim ready after {time.time() - now:.2f} seconds.")

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        isaacsim_path: Path,
    ):
        super().__init__(input_dir, output_dir)
        self._isaacsim_path = isaacsim_path.expanduser().resolve()
        self._process = None

    def convert(self, input_file: str, output_file: str, retries: int = 3) -> bool:
        input_full_path = self.input_dir / input_file
        if not input_full_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_full_path}")

        output_full_path = self.output_dir / output_file
        output_full_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Converting %s to %s...", input_full_path, output_full_path)

        cmd = f"{input_full_path}:{output_full_path}"

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
