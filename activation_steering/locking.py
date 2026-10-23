"""OS-owned file locks; process death releases ownership automatically."""

import os


def lock_file(stream, *, blocking=True):
    if os.name == "nt":
        import msvcrt
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(stream.fileno(), mode, 1)
        except OSError as error:
            raise BlockingIOError("Dateisperre wird bereits gehalten.") from error
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def unlock_file(stream):
    if os.name == "nt":
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream, fcntl.LOCK_UN)
