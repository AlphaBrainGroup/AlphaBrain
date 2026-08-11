from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from alphabrain_ui.gpu import GPUMonitor


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        *,
        device_count: int = 1,
        fail_memory_once: bool = False,
        failed_devices: set[int] | None = None,
        fail_device_count: bool = False,
    ) -> None:
        self.device_count = device_count
        self.fail_memory_once = fail_memory_once
        self.failed_devices = failed_devices or set()
        self.fail_device_count = fail_device_count
        self.init_count = 0
        self.shutdown_count = 0

    def nvmlInit(self) -> None:
        self.init_count += 1

    def nvmlShutdown(self) -> None:
        self.shutdown_count += 1

    def nvmlDeviceGetCount(self) -> int:
        if self.fail_device_count:
            raise RuntimeError("driver reset")
        return self.device_count

    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> SimpleNamespace:
        if self.fail_memory_once:
            self.fail_memory_once = False
            raise RuntimeError("Unknown Error")
        if handle in self.failed_devices:
            raise RuntimeError("device lost")
        return SimpleNamespace(total=4 * 1024**3, used=1024**3, free=3 * 1024**3)

    @staticmethod
    def nvmlDeviceGetUtilizationRates(_handle: int) -> SimpleNamespace:
        return SimpleNamespace(gpu=25)

    @staticmethod
    def nvmlDeviceGetComputeRunningProcesses(_handle: int) -> list[object]:
        return []

    @staticmethod
    def nvmlDeviceGetGraphicsRunningProcesses(_handle: int) -> list[object]:
        return []

    @staticmethod
    def nvmlDeviceGetTemperature(_handle: int, _sensor: int) -> int:
        return 52

    @staticmethod
    def nvmlDeviceGetName(handle: int) -> str:
        return f"Fake GPU {handle}"

    @staticmethod
    def nvmlDeviceGetUUID(handle: int) -> str:
        return f"GPU-{handle}"


def test_transient_device_failure_reinitializes_once_and_recovers() -> None:
    nvml = FakeNVML(fail_memory_once=True)
    monitor = GPUMonitor(nvml)

    items = monitor.snapshot()

    assert [(item.name, item.error, item.memory_total_bytes) for item in items] == [
        ("Fake GPU 0", "", 4 * 1024**3)
    ]
    assert monitor.available is True
    assert monitor.error == ""
    assert nvml.init_count == 2
    assert nvml.shutdown_count == 1


def test_persistent_device_failure_reports_stage_and_stays_unschedulable() -> None:
    nvml = FakeNVML(failed_devices={0})
    monitor = GPUMonitor(nvml)

    items = monitor.snapshot()

    assert len(items) == 1
    assert items[0].available is False
    assert "memory: device lost" in items[0].error
    assert monitor.available is False
    assert "GPU 0 memory: device lost" in monitor.error
    assert nvml.init_count == 2
    assert nvml.shutdown_count == 1


def test_partial_device_failure_keeps_healthy_devices_available() -> None:
    nvml = FakeNVML(device_count=2, failed_devices={1})
    monitor = GPUMonitor(nvml)

    items = monitor.snapshot({0: "job-1"})

    assert items[0].name == "Fake GPU 0"
    assert items[0].reserved_by_job_id == "job-1"
    assert items[0].available is False
    assert items[1].available is False
    assert "memory: device lost" in items[1].error
    assert monitor.available is True
    assert "GPU 1 memory: device lost" in monitor.error


def test_nvml_device_count_failure_retries_once_and_degrades() -> None:
    nvml = FakeNVML(fail_device_count=True)
    monitor = GPUMonitor(nvml)

    assert monitor.snapshot() == []
    assert monitor.available is False
    assert "device_count: driver reset" in monitor.error
    assert nvml.init_count == 2
    assert nvml.shutdown_count == 1


def test_concurrent_snapshots_and_close_share_one_nvml_lifecycle() -> None:
    nvml = FakeNVML()
    monitor = GPUMonitor(nvml)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(monitor.snapshot) for _ in range(8)]
        close_future = executor.submit(monitor.close)
        futures.extend(executor.submit(monitor.snapshot) for _ in range(24))
        snapshots = [future.result() for future in futures]
        close_future.result()

    assert all(not rows or rows[0].error == "" for rows in snapshots)
    assert nvml.init_count == 1
    assert nvml.shutdown_count == 1

    monitor.close()

    assert monitor.available is False
    assert monitor.snapshot() == []
    assert nvml.init_count == 1
    assert nvml.shutdown_count == 1
