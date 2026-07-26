import sys
from pathlib import Path
from unittest.mock import patch

import main


def test_ensure_runtime_dependencies_installs_when_missing(tmp_path):
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("litellm\n", encoding="utf-8")

    def fake_find_spec(module_name):
        if module_name == "litellm":
            return None
        return object()

    with patch("main.importlib.util.find_spec", side_effect=fake_find_spec):
        with patch("main.subprocess.check_call") as check_call:
            main._ensure_runtime_dependencies(requirements_file)

    check_call.assert_called_once_with([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_file),
    ])
