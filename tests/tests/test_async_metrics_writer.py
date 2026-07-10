import threading

import pytest

from ml.training.pretrain.engine import AsyncJsonlWriter


class _BlockingThreadCheckedFile:
    def __init__(self, *, allow_write: threading.Event, main_thread_id: int) -> None:
        self._allow_write = allow_write
        self._main_thread_id = int(main_thread_id)
        self.write_started = threading.Event()
        self.closed = False
        self.written: list[str] = []

    def write(self, s: str) -> int:
        if int(threading.get_ident()) == int(self._main_thread_id):
            raise AssertionError("sync write from main thread (queue overflow fallback)")
        self.write_started.set()
        self._allow_write.wait(timeout=10.0)
        self.written.append(str(s))
        return len(s)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FailingFile:
    def __init__(self) -> None:
        self.closed = False
        self.write_started = threading.Event()

    def write(self, _s: str) -> int:
        self.write_started.set()
        raise OSError("disk full")

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_async_jsonl_writer_raises_on_overflow_without_sync_write() -> None:
    allow_write = threading.Event()
    fp = _BlockingThreadCheckedFile(
        allow_write=allow_write, main_thread_id=threading.get_ident()
    )
    writer = AsyncJsonlWriter(fp, max_queue=1)
    try:
        writer.write_line("0\n")
        assert fp.write_started.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="queue is full"):
            for i in range(200):
                writer.write_line(f"{i}\n")
    finally:
        allow_write.set()
        writer.close()

    assert fp.closed


def test_async_jsonl_writer_close_raises_background_write_error() -> None:
    fp = _FailingFile()
    writer = AsyncJsonlWriter(fp, max_queue=1)

    writer.write_line("x\n")
    assert fp.write_started.wait(timeout=2.0)

    with pytest.raises(RuntimeError, match="Async metrics writer failed"):
        writer.close()

    assert fp.closed
