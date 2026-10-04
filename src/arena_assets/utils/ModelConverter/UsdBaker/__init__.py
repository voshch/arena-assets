import abc
import select
import subprocess
from pathlib import Path

from arena_assets.utils import logging


class UsdBaker(abc.ABC):
    """Handle USD baking."""

    _process: subprocess.Popen | None = None

    def write(self, msg: bytes) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("Interactive Python session not started")
        self._process.stdin.write(msg)
        self._process.stdin.flush()

    def readline(self) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("Interactive Python session not started")

        try:
            stdout = self._process.stdout
            stdout_fd = stdout.fileno()
            while True:
                if self._process.poll() is not None:
                    return None

                ready, _, _ = select.select([stdout_fd], [], [], 0.1)
                if not ready:
                    continue

                line = stdout.readline()
                if line:
                    return line
                if self._process.poll() is not None:
                    return None
        except (BrokenPipeError, OSError):
            return None

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir: Path = Path(input_dir).resolve()
        self.output_dir: Path = Path(output_dir).resolve()
        self.logger = logging.get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def start(self): ...

    def cleanup(self):
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None

    @classmethod
    def converter_script(cls) -> Path:
        relpath = Path(__file__).parent / "converter.py"
        return relpath.resolve()

    def __del__(self):
        """Destructor - cleanup when object is garbage collected."""
        try:
            self.cleanup()
        except Exception:
            pass

    def wait_for_ready(self) -> str:
        while True:
            bline = self.readline()
            if not bline:
                continue
            line = bline.decode("utf-8").rstrip()
            self.logger.debug("%s", line)
            if line.startswith("ready:"):
                self.logger.info("USD Baker is ready.")
                return line[len("ready:") :]

    def command(self, cmd: str) -> str | None:
        self.write(cmd.encode("utf-8") + b"\n")

        while (bline := self.readline()) is not None:
            if not bline:
                continue
            line = bline.decode("utf-8").rstrip()
            self.logger.info("%s", line)
            if line.startswith("success:"):
                return line[len("success:") :]
            if line.startswith("error:"):
                raise RuntimeError(line[len("error:") :])

        return None

    @abc.abstractmethod
    def convert(self, input_file: str, output_file: str, retries: int = 3) -> bool:
        raise NotImplementedError()
