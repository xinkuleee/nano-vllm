from nanovllm.benchmarks import gpu_environment


def test_collect_environment_reports_missing_gpu_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gpu_environment.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        gpu_environment,
        "_package_versions",
        lambda: {package: None for package in gpu_environment.REQUIRED_PACKAGES},
    )
    monkeypatch.setattr(
        gpu_environment,
        "_torch_report",
        lambda: ({"import_error": "missing"}, [
            gpu_environment.CheckResult("torch_import", "fail", "missing")
        ]),
    )
    monkeypatch.setattr(
        gpu_environment,
        "_import_checks",
        lambda: [gpu_environment.CheckResult("import:triton", "fail", "missing")],
    )

    report = gpu_environment.collect_environment(tmp_path)

    assert report["ready"] is False
    assert report["system"]["platform"] == "Darwin"
    assert report["packages"]["flash-attn"] is None
    assert any(check["name"] == "platform" for check in report["checks"])


def test_write_json_report_creates_parent_directory(tmp_path):
    output = tmp_path / "nested" / "environment.json"

    result = gpu_environment.write_json_report({"ready": True}, output)

    assert result == output
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_flash_attention_runtime_check_reports_failure_without_cuda():
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    result = gpu_environment._flash_attention_runtime_check(FakeTorch())

    assert result.status == "fail"
    assert "CUDA" in result.detail


def test_collect_environment_reports_ready_linux_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_environment.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        gpu_environment,
        "_package_versions",
        lambda: {package: "test" for package in gpu_environment.REQUIRED_PACKAGES},
    )
    monkeypatch.setattr(
        gpu_environment,
        "_torch_report",
        lambda: ({"cuda_available": True}, [
            gpu_environment.CheckResult("cuda_available", "pass", "available"),
            gpu_environment.CheckResult("compute_capability", "pass", "(8, 6)"),
            gpu_environment.CheckResult("cuda_allocation", "pass", "ok"),
        ]),
    )
    monkeypatch.setattr(gpu_environment.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        gpu_environment,
        "_run",
        lambda command: {"command": command, "returncode": 0, "stdout": "ok", "stderr": ""},
    )
    monkeypatch.setattr(
        gpu_environment,
        "_import_checks",
        lambda: [
            gpu_environment.CheckResult(f"import:{package}", "pass", module)
            for package, module in gpu_environment.REQUIRED_IMPORTS.items()
        ],
    )

    report = gpu_environment.collect_environment(tmp_path)

    assert report["ready"] is True


def test_import_checks_catch_broken_optional_extension(monkeypatch):
    def import_module(name):
        if name == "flash_attn":
            raise OSError("undefined symbol")
        return object()

    monkeypatch.setattr(gpu_environment.importlib, "import_module", import_module)

    checks = {check.name: check for check in gpu_environment._import_checks()}

    assert checks["import:triton"].status == "pass"
    assert checks["import:flash-attn"].status == "fail"
    assert "undefined symbol" in checks["import:flash-attn"].detail
