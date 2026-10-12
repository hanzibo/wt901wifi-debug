"""Launcher/tool checks are isolated: never open GUI, sockets or simulators."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_project, check_repo, run_smoke

ROOT = Path(__file__).resolve().parents[1]


class RepositoryToolsTests(unittest.TestCase):
    def test_missing_smoke_pattern_is_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_smoke, 'ROOT', Path(directory)), \
                patch.object(sys, 'path', sys.path.copy()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_smoke.main(), 1)

    def test_every_registered_entry_exists(self):
        for entry in run_project.ENTRIES:
            command = run_project.launch_command(ROOT, entry, ["--help"])
            self.assertTrue(Path(command[-2]).is_file())
            self.assertEqual(command[-1], "--help")

    def test_conda_selection_and_argument_preservation(self):
        command = run_project.launch_command(ROOT, "body", ["--replay", "my file.csv"], "demo")
        self.assertEqual(command[:6], ["conda", "run", "--no-capture-output", "-n", "demo", "python"])
        self.assertEqual(command[-2:], ["--replay", "my file.csv"])

    def test_list_and_dry_run_never_start_process(self):
        for args in (["--list"], ["--dry-run", "body", "--", "--help"]):
            with patch.object(run_project.subprocess, "run") as call, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_project.main(args), 0)
                call.assert_not_called()

    def test_launch_environment_and_exit_code(self):
        with patch.object(run_project.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as call:
            self.assertEqual(run_project.main(["body", "--", "--help"]), 7)
        self.assertIn(str(ROOT / "OptUSBProj-ubuntu"), call.call_args.kwargs['env']['PYTHONPATH'])
        self.assertNotIn('cwd', call.call_args.kwargs)

    def test_missing_entry_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_project.launch_command(Path(directory), "body", [])

    def test_unavailable_environment_is_failure(self):
        with patch.object(run_project.subprocess, "run", side_effect=OSError("missing conda")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_project.main(["--env", "missing", "body"]), 1)

    def test_external_cwd_real_help_without_gui_import(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_project.py"),
                                     "body", "--", "--help"], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--listen-port', result.stdout)

    def test_launch_keeps_user_relative_paths(self):
        with patch.object(run_project.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as call:
            run_project.main(["body", "--", "--settings", "relative.json"])
        self.assertEqual(call.call_args.args[0][-2:], ['--settings', 'relative.json'])

    def test_syntax_errors_and_missing_entries_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Test scratch files, never application configuration.
            (root / 'broken.py').write_text('def broken(:\n', encoding='utf-8')
            responses = [subprocess.CompletedProcess([], 0, stdout=b'broken.py\0'),
                         subprocess.CompletedProcess([], 0, stdout='', stderr='')]
            with patch.object(check_repo.subprocess, 'run', side_effect=responses):
                count, large, errors = check_repo.inspect(root)
            self.assertEqual(count, 1)
            self.assertFalse(large)
            self.assertTrue(any('broken.py' in e for e in errors))
            self.assertTrue(any('Missing entry' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
