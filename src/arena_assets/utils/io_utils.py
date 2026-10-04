import contextlib
import io
import os
import sys
import threading

# The pty module is only available on POSIX-compliant systems (Linux, macOS).
if os.name == "posix":
    import pty
else:
    pty = None


def _fd_reader(read_fd: int, buffer: io.IOBase):
    """
    Reads data from a file descriptor (like a pipe or PTY) in a loop and
    writes it to a buffer. This is intended to be run in a separate thread.

    Args:
        read_fd (int): The file descriptor for the reading end.
        buffer (io.IOBase): The buffer to write the captured output to.
    """
    while True:
        try:
            # Read up to 1024 bytes.
            data = os.read(read_fd, 1024)
            if not data:
                # An empty byte string means the write-end was closed.
                break
            # Decode and write to the provided in-memory buffer.
            buffer.write(data.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            # This can happen if the fd or buffer is closed unexpectedly.
            break


@contextlib.contextmanager
def capture_all_output(stdout: io.IOBase | None = None, stderr: io.IOBase | None = None):
    """
    A context manager that captures all C-level and Python output by redirecting
    it to a pseudo-terminal (PTY), ensuring real-time, unbuffered capture.

    NOTE: This advanced implementation is only available on Linux and macOS.

    Args:
        stdout (io.IOBase, optional): The buffer to which stdout will be appended.
        stderr (io.IOBase, optional): The buffer to which stderr will be appended.

    Yields:
        (io.IOBase, io.IOBase): A tuple with the stdout and stderr buffers.
    """
    if pty is None:
        raise NotImplementedError("The pty-based output capturer is only available on Linux or macOS.")

    if stdout is None:
        stdout = io.StringIO()
    if stderr is None:
        stderr = io.StringIO()

    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Create pseudo-terminals. pty.openpty() returns a (master, slave) pair.
    # We write to the slave, and read from the master.
    stdout_master_fd, stdout_slave_fd = pty.openpty()
    stderr_master_fd, stderr_slave_fd = pty.openpty()

    # Start reader threads that listen on the MASTER ends of the PTYs.
    stdout_thread = threading.Thread(target=_fd_reader, args=(stdout_master_fd, stdout))
    stderr_thread = threading.Thread(target=_fd_reader, args=(stderr_master_fd, stderr))
    stdout_thread.start()
    stderr_thread.start()

    try:
        # Redirect stdout/stderr to the SLAVE ends of the PTYs.
        # This fools the C-library into thinking it's writing to a real terminal.
        os.dup2(stdout_slave_fd, original_stdout_fd)
        os.dup2(stderr_slave_fd, original_stderr_fd)

        yield stdout, stderr
    finally:
        # Flush Python's buffers to ensure everything is sent to the PTYs.
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore the original file descriptors.
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)

        # Close the slave ends. This sends an EOF to the master ends,
        # causing the reader threads to terminate their loops.
        os.close(stdout_slave_fd)
        os.close(stderr_slave_fd)

        # Wait for the reader threads to finish processing all data.
        stdout_thread.join()
        stderr_thread.join()

        # Clean up all remaining file descriptors.
        os.close(stdout_master_fd)
        os.close(stderr_master_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
