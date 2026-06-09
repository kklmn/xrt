# -*- coding: utf-8 -*-
"""Run raycing example scripts with smoke-test tracing limits.

This helper runs every selected example in a separate Python process. In the
child process it patches :func:`xrt.runner.run_ray_tracing` so that scripted
examples use a small, fixed number of rays and repeats while keeping each
example's own initialization and generator setup code.

Typical use from the repository root::

    python tests/run_raycing_examples.py --nrays 10000 --repeats 1

Use ``--list`` first when tuning include/exclude filters.
"""

from __future__ import print_function

import argparse
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = ROOT / "examples" / "withRaycing"


class _ExampleSourceTransformer(ast.NodeTransformer):
    """Small in-memory tweaks for examples that default to GUI mode."""

    def __init__(self, force_no_3d=True, force_rays=False):
        self.force_no_3d = force_no_3d
        self.force_rays = force_rays

    def visit_Module(self, node):
        new_body = []
        for stmt in node.body:
            if self.force_no_3d and _assigns_name(stmt, "showIn3D"):
                stmt = _replace_assignment_value(stmt, ast.Constant(False))
            elif self.force_rays and _assigns_name(stmt, "what"):
                stmt = _replace_assignment_value(stmt, ast.Constant("rays"))
            new_body.append(stmt)
        node.body = new_body
        ast.fix_missing_locations(node)
        return node


def _assigns_name(stmt, name):
    if isinstance(stmt, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name
                   for target in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == name
    return False


def _replace_assignment_value(stmt, value):
    if isinstance(stmt, ast.Assign):
        return ast.Assign(targets=stmt.targets, value=value)
    if isinstance(stmt, ast.AnnAssign):
        return ast.AnnAssign(
            target=stmt.target, annotation=stmt.annotation, value=value,
            simple=stmt.simple)
    return stmt


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _display_path(path, root=ROOT):
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def _iter_example_scripts(paths, repo_root):
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(path.rglob("*.py"))
        for candidate in candidates:
            if candidate.name == "__init__.py":
                continue
            if "__pycache__" in candidate.parts:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _matches_filters(path, repo_root, only, exclude, exclude_dirs):
    rel = _display_path(path, repo_root)
    rel_posix = rel.replace(os.sep, "/")
    if only and not any(token in rel_posix for token in only):
        return False
    if exclude and any(token in rel_posix for token in exclude):
        return False
    parts = set(Path(rel).parts)
    if exclude_dirs and any(dirname in parts for dirname in exclude_dirs):
        return False
    return True


def _safe_log_name(script, repo_root):
    rel = _display_path(script, repo_root)
    name = rel.replace("\\", "__").replace("/", "__").replace(":", "")
    return name


def _patch_gui_entry_points():
    try:
        from xrt.backends.raycing import beamline
    except Exception:
        return

    def _skip_gui(self, *args, **kwargs):
        print("xrt example runner: skipped GUI call on {0}".format(
            getattr(self, "name", self.__class__.__name__)))
        return None

    beamline.BeamLine.glow = _skip_gui
    beamline.BeamLine.explore = _skip_gui


def _patch_matplotlib(keep_artifacts):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    def _show(*args, **kwargs):
        plt.close("all")

    plt.show = _show

    if not keep_artifacts:
        from matplotlib.figure import Figure

        def _skip_savefig(self, *args, **kwargs):
            if args:
                print("xrt example runner: skipped savefig {0}".format(
                    args[0]))
            return None

        Figure.savefig = _skip_savefig


def _set_beamline_nrays(beam_line, nrays):
    if beam_line is None:
        return
    if hasattr(beam_line, "nrays"):
        beam_line.nrays = nrays
    for source in getattr(beam_line, "sources", []):
        if hasattr(source, "nrays"):
            source.nrays = nrays


def _suppress_plot_artifacts(plots):
    for plot in _as_list(plots):
        if hasattr(plot, "saveName"):
            plot.saveName = None
        if hasattr(plot, "persistentName"):
            plot.persistentName = None


def _limited_generator_factory(generator, max_steps):
    def _limited_generator(*args, **kwargs):
        generated = generator(*args, **kwargs)
        if generated is None:
            return
        for step, value in enumerate(generated):
            if step >= max_steps:
                break
            yield value

    _limited_generator.__name__ = getattr(
        generator, "__name__", "_limited_generator")
    return _limited_generator


def _patch_runner(nrays, repeats, generator_steps, keep_artifacts):
    from xrt.backends import raycing
    import xrt.runner as xrtr

    raycing.nrays = nrays
    original_run_ray_tracing = xrtr.run_ray_tracing
    signature = inspect.signature(original_run_ray_tracing)

    def _run_ray_tracing_smoke(*args, **kwargs):
        caller_frame = inspect.currentframe().f_back
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        plots = bound.arguments.get("plots")
        beam_line = bound.arguments.get("beamLine")
        _set_beamline_nrays(beam_line, nrays)
        if not keep_artifacts:
            _suppress_plot_artifacts(plots)

        bound.arguments["repeats"] = repeats
        bound.arguments["updateEvery"] = 1
        bound.arguments["threads"] = 1
        bound.arguments["processes"] = 1
        bound.arguments["pickleEvery"] = None
        bound.arguments["globalNorm"] = 0
        bound.arguments["afterScript"] = None

        generator = bound.arguments.get("generator")
        if generator is not None:
            generator_args = bound.arguments.get("generatorArgs") or []
            generator_kwargs = bound.arguments.get("generatorKWargs")
            if generator_kwargs == "auto":
                generator_name = getattr(generator, "__name__", "")
                if (generator_name in caller_frame.f_locals) or generator_args:
                    generator_kwargs = {}
                else:
                    generator_kwargs = {"plots": plots, "beamLine": beam_line}
                bound.arguments["generatorKWargs"] = generator_kwargs
            if generator_steps is not None:
                bound.arguments["generator"] = _limited_generator_factory(
                    generator, generator_steps)

        return original_run_ray_tracing(**bound.arguments)

    xrtr.run_ray_tracing = _run_ray_tracing_smoke


def _execute_script(script, repo_root, force_no_3d, force_rays):
    script = Path(script).resolve()
    os.chdir(str(script.parent))
    for path in (str(repo_root), str(script.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)

    old_argv = sys.argv[:]
    sys.argv = [str(script)]
    try:
        source = script.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(script))
        tree = _ExampleSourceTransformer(
            force_no_3d=force_no_3d, force_rays=force_rays).visit(tree)
        globals_dict = {
            "__name__": "__main__",
            "__file__": str(script),
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(tree, str(script), "exec"), globals_dict)
    finally:
        sys.argv = old_argv


def run_one(args):
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["XRT_EXAMPLE_NRAYS"] = str(args.nrays)
    os.environ["XRT_EXAMPLE_REPEATS"] = str(args.repeats)

    _patch_matplotlib(args.keep_artifacts)
    _patch_gui_entry_points()
    _patch_runner(
        args.nrays, args.repeats,
        None if args.generator_steps < 1 else args.generator_steps,
        args.keep_artifacts)
    _execute_script(
        args.run_one, repo_root, args.force_no_3d, args.force_rays)
    return 0


def run_suite(args):
    repo_root = Path(args.repo_root).resolve()
    paths = args.paths or [str(DEFAULT_EXAMPLES)]
    exclude_dirs = list(args.exclude_dir)
    if args.skip_waves and "11_Waves" not in exclude_dirs:
        exclude_dirs.append("11_Waves")

    scripts = [
        script for script in _iter_example_scripts(paths, repo_root)
        if _matches_filters(
            script, repo_root, args.only, args.exclude, exclude_dirs)]

    if args.list:
        for script in scripts:
            print(_display_path(script, repo_root))
        print("{0} script(s)".format(len(scripts)))
        return 0

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Running {0} script(s); logs: {1}".format(len(scripts), log_dir))

    failures = []
    for index, script in enumerate(scripts, 1):
        rel = _display_path(script, repo_root)
        log_base = log_dir / _safe_log_name(script, repo_root)
        stdout_path = log_base.with_suffix(".out.log")
        stderr_path = log_base.with_suffix(".err.log")
        cmd = [
            args.python,
            str(Path(__file__).resolve()),
            "--run-one", str(script),
            "--repo-root", str(repo_root),
            "--nrays", str(args.nrays),
            "--repeats", str(args.repeats),
            "--generator-steps", str(args.generator_steps),
        ]
        if args.keep_artifacts:
            cmd.append("--keep-artifacts")
        if args.no_force_no_3d:
            cmd.append("--no-force-no-3d")
        if args.force_rays:
            cmd.append("--force-rays")

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root), env.get("PYTHONPATH", "")])

        print("[{0}/{1}] {2}".format(index, len(scripts), rel))
        start = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, \
                stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.Popen(
                cmd, cwd=str(repo_root), env=env, stdout=stdout,
                stderr=stderr, text=True)
            try:
                proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                stderr.write("\nTIMEOUT after {0} s\n".format(args.timeout))
                rc = 124
            else:
                rc = proc.returncode
        elapsed = time.time() - start
        if rc:
            failures.append((rel, rc, stdout_path, stderr_path))
            print("  failed: exit={0}, {1:.1f} s".format(rc, elapsed))
            if args.fail_fast:
                break
        else:
            print("  ok: {0:.1f} s".format(elapsed))

    if failures:
        print("\nFailures:")
        for rel, rc, stdout_path, stderr_path in failures:
            print("  {0}: exit={1}".format(rel, rc))
            print("    stdout: {0}".format(stdout_path))
            print("    stderr: {0}".format(stderr_path))
        return 1

    print("\nAll selected examples passed.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run raycing examples with reduced ray-tracing settings.")
    parser.add_argument(
        "paths", nargs="*",
        help="Example files or directories. Defaults to examples/withRaycing.")
    parser.add_argument(
        "--repo-root", default=str(ROOT),
        help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-one", help=argparse.SUPPRESS)
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python executable for child processes. Default: current Python.")
    parser.add_argument(
        "--nrays", type=int, default=10000,
        help="Number of rays assigned to raycing sources. Default: 10000.")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="run_ray_tracing repeats value. Default: 1.")
    parser.add_argument(
        "--generator-steps", type=int, default=1,
        help="Maximum generator scan steps. Use 0 for unlimited. Default: 1.")
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per script in seconds. Default: 300.")
    parser.add_argument(
        "--log-dir",
        default=str(Path(tempfile.gettempdir()) / "xrt-example-logs"),
        help="Directory for child stdout/stderr logs.")
    parser.add_argument(
        "--only", action="append", default=[],
        help="Run scripts whose repository-relative path contains this text.")
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="Skip scripts whose repository-relative path contains this text.")
    parser.add_argument(
        "--exclude-dir", action="append", default=[],
        help="Skip scripts below a directory with this exact name.")
    parser.add_argument(
        "--skip-waves", action="store_true",
        help="Shortcut for --exclude-dir 11_Waves.")
    parser.add_argument(
        "--force-rays", action="store_true",
        help="Set module-level what = 'rays' in memory before execution.")
    parser.add_argument(
        "--no-force-no-3d", dest="no_force_no_3d", action="store_true",
        help="Do not force module-level showIn3D = False.")
    parser.add_argument(
        "--keep-artifacts", action="store_true",
        help="Allow examples to save figures/persistent files.")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop after the first failure.")
    parser.add_argument(
        "--list", action="store_true",
        help="List selected scripts without running them.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_one:
        args.force_no_3d = not args.no_force_no_3d
        return run_one(args)
    return run_suite(args)


if __name__ == "__main__":
    sys.exit(main())
