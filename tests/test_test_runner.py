"""Test entry-point orchestration without running nested suites."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "scripts/run_tests.py"
SPEC = importlib.util.spec_from_file_location("wt901_test_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TestRunnerTests(unittest.TestCase):
    def run_entry(self, args, codes):
        from types import SimpleNamespace
        results = [SimpleNamespace(returncode=code) for code in codes]
        with patch.object(sys, "argv", [str(PATH), *args]), \
                patch.object(runner.subprocess, "run", side_effect=results) as call, \
                patch("builtins.print"):
            code = runner.main()
        return code, call.call_args_list

    def test_all_runs_separate_suites_and_propagates_failure(self):
        code, calls = self.run_entry([], [1, 0])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 2)
        self.assertIn(str(runner.ROOT / "tests"), calls[0].args[0])
        self.assertIn(str(runner.ROOT / "OptUSBProj-ubuntu/tests"), calls[1].args[0])

    def test_scene_paths_and_environment_do_not_depend_on_cwd(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/existing", "QT_QPA_PLATFORM": "offscreen"}):
            code, calls = self.run_entry(["--suite", "scene"], [0])
        self.assertEqual(code, 0)
        command = calls[0].args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("test_scene*.py", command)
        options = calls[0].kwargs
        self.assertEqual(options["cwd"], runner.ROOT)
        self.assertEqual(options["env"]["QT_QPA_PLATFORM"], "offscreen")
        self.assertEqual(options["env"]["PYTHONPATH"].split(os.pathsep),
                         [str(runner.ROOT), str(runner.ROOT / "OptUSBProj-ubuntu"), "/existing"])

    def test_pattern_and_verbose_are_forwarded(self):
        code, calls = self.run_entry(["--suite", "app", "--pattern", "test_pose.py", "--verbose"], [0])
        self.assertEqual(code, 0)
        self.assertIn("test_pose.py", calls[0].args[0])
        self.assertEqual(calls[0].args[0][-1], "-v")

    def test_each_suite_can_select_its_existing_conda_environment(self):
        code, calls = self.run_entry(["--core-env", "wt901", "--app-env", "OptUSBProj-yolo"], [0, 0])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0].args[0][:6],
                         ["conda", "run", "--no-capture-output", "-n", "wt901", "python"])
        self.assertEqual(calls[1].args[0][:6],
                         ["conda", "run", "--no-capture-output", "-n", "OptUSBProj-yolo", "python"])

    def test_missing_executable_is_failure_not_silent_skip(self):
        with patch.object(sys, "argv", [str(PATH), "--suite", "core"]), \
                patch.object(runner.subprocess, "run", side_effect=FileNotFoundError("conda")), \
                patch("builtins.print"):
            self.assertEqual(runner.main(), 1)
