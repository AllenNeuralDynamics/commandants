"""Command-line interface: ``commandants <subcommand>``.

Subcommands
-----------
install-ants     download prebuilt ANTs binaries into the managed directory
install-elastix  download prebuilt elastix binaries into the managed directory
which            print the resolved path of a binary (ANTs or elastix/transformix)
version          print commandants, ANTs, and elastix versions
list             list managed installs (--tool ants|elastix|all)
uninstall-ants   remove a managed ANTs install
uninstall-elastix  remove a managed elastix install
info             show the managed data directory
explain          explain a process exit code (e.g. -9, 137)
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core.exceptions import CommandantsError


def _cmd_install_ants(args: argparse.Namespace) -> int:
    from .install import install_ants

    bindir = install_ants(
        version=args.version,
        dest=args.dest,
        asset=args.asset,
        force=args.force,
        quiet=args.quiet,
    )
    print(bindir)
    return 0


def _cmd_which(args: argparse.Namespace) -> int:
    from .core.executable import resolve_binary

    print(resolve_binary(args.name, auto_install=args.auto_install))
    return 0


def _cmd_install_elastix(args: argparse.Namespace) -> int:
    from .install import install_elastix

    bindir = install_elastix(
        version=args.version,
        dest=args.dest,
        asset=args.asset,
        force=args.force,
        quiet=args.quiet,
    )
    print(bindir)
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    from .core.executable import is_available, version

    print(f"commandants {__version__}")
    if is_available("antsRegistration"):
        print(version("antsRegistration"))  # self-identifies, e.g. "ANTs Version: 2.6.5-..."
    else:
        print("ANTs: not found (run `commandants install-ants`)")
    if is_available("elastix"):
        print(version("elastix").splitlines()[0])
    else:
        print("elastix: not found (run `commandants install-elastix`)")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .install import installed_elastix_versions, installed_versions

    tool = getattr(args, "tool", "all")
    printed = False
    if tool in ("ants", "all"):
        for ver, bindir in sorted(installed_versions().items()):
            if not printed:
                print("ANTs:")
            print(f"  {ver}\t{bindir}")
            printed = True
    if tool in ("elastix", "all"):
        elastix = sorted(installed_elastix_versions().items())
        if elastix:
            print("elastix:")
        for ver, bindir in elastix:
            print(f"  {ver}\t{bindir}")
            printed = True
    if not printed:
        print("No managed installs. Run `commandants install-ants` / `install-elastix`.")
    return 0


def _cmd_uninstall_ants(args: argparse.Namespace) -> int:
    from .install import uninstall_ants

    removed = uninstall_ants(version=args.version)
    if not removed:
        print("Nothing to remove.")
    for path in removed:
        print(f"removed {path}")
    return 0


def _cmd_uninstall_elastix(args: argparse.Namespace) -> int:
    from .install import uninstall_elastix

    removed = uninstall_elastix(version=args.version)
    if not removed:
        print("Nothing to remove.")
    for path in removed:
        print(f"removed {path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from .install import user_data_dir

    print(user_data_dir())
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .exit_codes import explain_exit_code

    print(explain_exit_code(int(args.code)).text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commandants", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"commandants {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser("install-ants", help="download prebuilt ANTs binaries")
    p_install.add_argument("--version", default="2.6.5", help="ANTs version or 'latest' (default: 2.6.5)")
    p_install.add_argument("--asset", default=None, help="explicit release asset name (override auto-select)")
    p_install.add_argument("--dest", default=None, help="install root (default: managed data dir)")
    p_install.add_argument("--force", action="store_true", help="re-download even if present")
    p_install.add_argument("--quiet", action="store_true", help="suppress progress output")
    p_install.set_defaults(func=_cmd_install_ants)

    p_install_e = sub.add_parser("install-elastix", help="download prebuilt elastix binaries")
    p_install_e.add_argument("--version", default="5.3.1", help="elastix version or 'latest' (default: 5.3.1)")
    p_install_e.add_argument("--asset", default=None, help="explicit release asset name (override auto-select)")
    p_install_e.add_argument("--dest", default=None, help="install root (default: managed data dir)")
    p_install_e.add_argument("--force", action="store_true", help="re-download even if present")
    p_install_e.add_argument("--quiet", action="store_true", help="suppress progress output")
    p_install_e.set_defaults(func=_cmd_install_elastix)

    p_which = sub.add_parser("which", help="print the resolved path of a binary (ANTs or elastix)")
    p_which.add_argument("name", nargs="?", default="antsRegistration")
    p_which.add_argument(
        "--auto-install", dest="auto_install", action="store_true", help="download managed ANTs if not found"
    )
    p_which.set_defaults(func=_cmd_which)

    p_version = sub.add_parser("version", help="print commandants and ANTs versions")
    p_version.set_defaults(func=_cmd_version)

    p_list = sub.add_parser("list", help="list managed installs")
    p_list.add_argument(
        "--tool", choices=["ants", "elastix", "all"], default="all", help="which tool's installs to list (default: all)"
    )
    p_list.set_defaults(func=_cmd_list)

    p_uninstall = sub.add_parser("uninstall-ants", help="remove a managed ANTs install")
    p_uninstall.add_argument("--version", default=None, help="version to remove (default: all)")
    p_uninstall.set_defaults(func=_cmd_uninstall_ants)

    p_uninstall_e = sub.add_parser("uninstall-elastix", help="remove a managed elastix install")
    p_uninstall_e.add_argument("--version", default=None, help="version to remove (default: all)")
    p_uninstall_e.set_defaults(func=_cmd_uninstall_elastix)

    p_info = sub.add_parser("info", help="show the managed data directory")
    p_info.set_defaults(func=_cmd_info)

    p_explain = sub.add_parser("explain", help="explain a process exit code (e.g. -9, 137)")
    p_explain.add_argument("code", help="the exit/return code, e.g. -9 or 137")
    p_explain.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CommandantsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
