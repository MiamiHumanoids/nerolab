import os
import select
import sys


def read_char() -> str:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "No real terminal is available. Run this script in a system terminal."
        )

    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        while True:
            readable, _, _ = select.select([file_descriptor], [], [], 0.1)
            if readable:
                value = sys.stdin.buffer.read(1)
                if value:
                    return value.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)


def read_key_nonblocking() -> int:
    if not sys.stdin.isatty():
        return -1

    if os.name == "nt":
        import msvcrt

        return ord(msvcrt.getwch()) if msvcrt.kbhit() else -1

    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        readable, _, _ = select.select([file_descriptor], [], [], 0)
        return ord(sys.stdin.read(1)) if readable else -1
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)