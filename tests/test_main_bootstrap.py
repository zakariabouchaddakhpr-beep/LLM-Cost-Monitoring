import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import main


def test_ensure_runtime_dependencies_installs_when_missing(tmp_path):
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("litellm\n", encoding="utf-8")

    def fake_import_module(module_name):
        if module_name == "litellm.types.integrations.datadog":
            raise ImportError()
        return object()

    with patch.object(main.importlib, "import_module", side_effect=fake_import_module):
        with patch.object(main.subprocess, "check_call") as check_call:
            main._ensure_runtime_dependencies(requirements_file)

    check_call.assert_called_once_with([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
    ])


def test_ensure_runtime_dependencies_retries_force_reinstall_on_failure(tmp_path):
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("litellm\n", encoding="utf-8")

    def fake_import_module(module_name):
        if module_name == "litellm.types.integrations.datadog":
            raise ImportError()
        return object()

    with patch.object(main.importlib, "import_module", side_effect=fake_import_module):
        with patch.object(main.subprocess, "check_call") as mock_check_call:
            mock_check_call.side_effect = [subprocess.CalledProcessError(1, ["pip"]), None]
            main._ensure_runtime_dependencies(requirements_file)

    expected_calls = [
        ([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file),
        ],),
        ([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--ignore-installed",
            "--force-reinstall",
            "-r",
            str(requirements_file),
        ],),
    ]
    actual_calls = [call.args for call in mock_check_call.call_args_list]
    assert actual_calls == expected_calls
