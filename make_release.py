"""Assemble the distributable zip from an existing build under dist/.

ImageJ installs a plug-in by its position in the Fiji tree, so the zip is laid
out the way Fiji expects to find things -- plugins/ at the top -- and is meant
to be unpacked into the Fiji root, where ImageJ-win64.exe is:

    plugins/Auto_Worm_ROI.jar
    plugins/AutoWormImageJ/AutoWormGUI.exe
    plugins/AutoWormImageJ/{models,_internal,licenses}/...
    README_0.4.2_ImageJ_Java8.md  CHANGELOG_0.4.2.md  VALIDATION_0.4.2.md  LICENSE

Three checks run before anything is packed, because each one covers a mistake
that produces a zip which looks completely normal:

  * the version string agrees in all three places that carry it, so a partial
    bump cannot ship a jar that announces one version and a summary CSV that
    records another;
  * every module the EXE shares with src/ is the same code as src/ -- compares
    the packaged code objects against freshly compiled ones, module by module,
    rather than searching for a marker string somebody has to remember to add;
  * when --previous-exe is given, that older build must NOT match the current
    source. Without that half, a matching comparison proves nothing about
    whether the rebuild did anything at all.

Run it with the build interpreter, after build_0.4.2.bat:

    <build python> make_release.py [--previous-exe <old AutoWormGUI.exe>]
"""

from __future__ import print_function

import argparse
import hashlib
import marshal
import re
import shutil
import sys
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
GUI_DIR = DIST / "AutoWormImageJ"
SRC = ROOT / "src"
PLUGIN_CONFIG = ROOT / "plugin_src" / "plugins.config"
RELEASE_ROOT = ROOT / "release"

# Carried in the archive name and the readme file names.
VERSION_PATTERN = re.compile(r'"Auto Worm ROI ([0-9][^"]*)"')

# The whole of the ImageJ-side plug-in: everything else is the GUI runtime.
JAR_NAME = "Auto_Worm_ROI.jar"
PACKAGED_GUI_DIR = "AutoWormImageJ"

# The docs that sit at the top of the zip. The numeric release ("0.4.2") is
# what the file names carry, the full version ("0.4.2-beta") what the zip does.
DOC_NAMES = ("README_%s_ImageJ_Java8.md", "CHANGELOG_%s.md", "VALIDATION_%s.md")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_version():
    """The version ImageJ shows in its menu, which is the one users quote.

    Taken from plugins.config rather than repeated here: a release script that
    states the version itself is one more place to forget to bump.
    """
    text = PLUGIN_CONFIG.read_text(encoding="ascii")
    match = VERSION_PATTERN.search(text)
    if not match:
        raise SystemExit("no version found in %s" % PLUGIN_CONFIG)
    # ImageJ menu strings spell the beta marker with a space; the code and the
    # zip name use a hyphen.
    return match.group(1).strip().replace(" ", "-")


def check_version_agreement(version):
    """Every place the version appears must agree, or the zip is inconsistent."""
    numeric = version.split("-")[0]
    problems = []

    in_code = {
        "src/worm_roi_gui.py (APP_VERSION)":
            (SRC / "worm_roi_gui.py", 'APP_VERSION = "%s"' % version),
        "src/batch_worm_roi.py (SOFTWARE_VERSION)":
            (SRC / "batch_worm_roi.py", 'SOFTWARE_VERSION = "%s' % version),
    }
    for label, (path, expected) in in_code.items():
        if expected not in path.read_text(encoding="utf-8"):
            problems.append("%s does not contain %r" % (label, expected))

    docs = []
    for template in DOC_NAMES:
        name = template % numeric
        docs.append(name)
        if not (ROOT / name).is_file():
            problems.append("missing %s" % name)
    docs.append("LICENSE")
    if not (ROOT / "LICENSE").is_file():
        problems.append("missing LICENSE")

    print("version: %s (menu), docs named for %s" % (version, numeric))
    return problems, docs


def constant_key(value):
    """A stable rendering of one constant carried by a code object.

    Unordered containers have to be normalised. The order repr() prints a
    frozenset in is the order of its internal table, which depends on how that
    table was filled -- from source order when compiling, from stored order when
    read back out of a .pyc. An unchanged frozenset really does print two ways
    depending on where the code object came from, so printing it as-is would
    report a difference that is not one.

    Elements are rendered first and the sort is over those strings, because the
    elements themselves need not be comparable with each other.
    """
    if isinstance(value, (set, frozenset)):
        inner = ",".join(sorted(constant_key(item) for item in value))
        return "frozenset({%s})" % inner if isinstance(value, frozenset) \
            else "{%s}" % inner
    if isinstance(value, tuple):
        return "(%s)" % ",".join(constant_key(item) for item in value)
    return repr(value)


def code_surface(code, out):
    """A fingerprint of a code object and everything nested inside it.

    Built only from the parts that survive being written to a .pyc by
    PyInstaller and read back, so that an EXE and freshly compiled source can be
    compared at all.

    Names and string literals are not enough on their own: changing `x + 1` to
    `x - 99` leaves every name and every string untouched, and the release check
    would then call a rebuilt EXE identical to source it no longer matches. So
    the numbers a module compiles to, its argument layout and its bytecode are
    part of the fingerprint as well. Docstrings, and default arguments, come
    along for free -- both are constants.

    `out` is a list and the order matters, because a flat set of lines is blind
    to a whole class of edit. `a = 1; b = 2` and `a = 2; b = 1` compile to the
    same bytecode -- the constant table is what changed, not the instructions
    that index into it -- so once the two constants have been sorted into a set,
    nothing distinguishes the two programs. The same goes for two nested
    functions exchanging bodies: the same names, the same numbers, different
    code. Order is preserved here instead, and a nested code object is expanded
    at the position it occupies in co_consts, which is what ties each constant
    and each name to the code object it belongs to. Nothing is lost by it: the
    compiler emits co_consts, co_names and co_varnames in a fixed order for a
    fixed source, and marshalling keeps that order, so two fingerprints of
    unchanged code still come out equal. Only genuinely unordered values are
    normalised -- see constant_key().

    Line numbers are deliberately left out. They do survive marshalling, but the
    only edits they would catch are whitespace and comments, which cannot change
    what the program does, and including them would make the check depend on
    formatting.
    """
    out.append("code %s(%d,%d,%d) flags=%d" % (code.co_name, code.co_argcount,
                                               code.co_posonlyargcount,
                                               code.co_kwonlyargcount, code.co_flags))
    out.extend("name " + name for name in code.co_names)
    out.extend("var " + name for name in code.co_varnames)
    out.extend("free " + name for name in code.co_freevars)
    out.extend("cell " + name for name in code.co_cellvars)
    out.append("bytecode " + code.co_code.hex())
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            code_surface(value, out)
        else:
            out.append("const " + constant_key(value))
    return out


def packaged_code_objects(exe, wanted):
    """The compiled modules inside a built EXE, for the names in `wanted`.

    Most of the program lives in the embedded PYZ; the entry script does not,
    and is stored in the outer archive instead. Both are needed -- the entry
    script is worm_roi_gui, the file a user's bug report is most likely about.

    Only the requested names are read: the PYZ also holds entries that are not
    marshalled code at all, and asking for those fails rather than returning
    something to compare.
    """
    from PyInstaller.archive.readers import CArchiveReader

    archive = CArchiveReader(str(exe))
    pyz = archive.open_embedded_archive("PYZ.pyz")
    modules = {}
    for name in wanted:
        try:
            if name in pyz.toc:
                modules[name] = marshal.loads(pyz.extract(name, raw=True))
            else:
                modules[name] = marshal.loads(archive.extract(name))
        except Exception:
            # Not compiled Python in this build. Reported as "not in this
            # build" by the caller rather than as a mismatch.
            continue
    return modules


def compare_exe_to_source(exe):
    """Modules present both in the EXE and in src/, and whether they match.

    Returns (matching, differing, missing) module-name lists. A module only in
    one of the two is not a mismatch -- PyInstaller pulls in third-party code
    and a src/ file nobody imported is simply absent -- so only the shared
    names are compared.
    """
    sources = {path.stem: path for path in sorted(SRC.glob("*.py"))}
    modules = packaged_code_objects(exe, set(sources))
    matching, differing, missing = [], [], []
    for name, path in sources.items():
        if name not in modules:
            missing.append(name)
            continue
        with open(path, "rb") as handle:
            fresh = compile(handle.read(), str(path), "exec")
        if code_surface(modules[name], []) == code_surface(fresh, []):
            matching.append(name)
        else:
            differing.append(name)
    return matching, differing, missing


def report_comparison(label, exe, verbose):
    if not Path(exe).is_file():
        raise SystemExit("no such EXE: %s" % exe)
    matching, differing, missing = compare_exe_to_source(exe)
    print("%s: %d module(s) match src/, %d differ"
          % (label, len(matching), len(differing)))
    if differing:
        print("    differ: %s" % ", ".join(differing))
    if missing:
        print("    not in this build: %s" % ", ".join(missing))
    if verbose:
        print("    match: %s" % ", ".join(matching))
    return matching, differing, missing


def sync_docs_to_dist(docs):
    """Refresh the doc copies under dist/.

    Nothing reads these -- the zip is packed straight from the repository root
    -- but dist/ is what a person browses when assembling a release by hand, and
    an out-of-date README sitting next to the jar is exactly the thing that gets
    shipped by mistake. Copying them here is cheap and removes the step from the
    checklist entirely.
    """
    print("Refreshing the doc copies under dist/")
    for name in docs:
        source = ROOT / name
        target = DIST / name
        if target.is_file() and sha256_file(target) == sha256_file(source):
            print("  %-26s unchanged" % name)
            continue
        shutil.copyfile(source, target)
        print("  %-26s updated" % name)


def collect_files(docs):
    """(path on disk, path inside the zip) for everything that ships."""
    entries = [(DIST / JAR_NAME, "plugins/" + JAR_NAME)]
    for path in sorted(GUI_DIR.rglob("*")):
        if path.is_file():
            relative = path.relative_to(GUI_DIR).as_posix()
            entries.append((path, "plugins/%s/%s" % (PACKAGED_GUI_DIR, relative)))
    for name in docs:
        entries.append((ROOT / name, name))
    return entries


def write_zip(entries, out_zip, compress):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    total = 0
    with zipfile.ZipFile(out_zip, "w", method, compresslevel=6) as archive:
        for index, (path, arcname) in enumerate(entries, 1):
            archive.write(path, arcname)
            total += path.stat().st_size
            if index % 200 == 0:
                print("  packed %d/%d files (%.0f MiB so far)"
                      % (index, len(entries), total / 1048576.0))
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Pack dist/ into the plugins/-rooted ImageJ zip.")
    parser.add_argument("--previous-exe", default=None,
                        help="the AutoWormGUI.exe of the build this one replaces. "
                             "The comparison must show it DIFFERENT from the "
                             "current src/, otherwise the matching check on the "
                             "new EXE proves nothing.")
    parser.add_argument("--outdir", default=None,
                        help="where to write the zip (default: release/ beside "
                             "the sources)")
    parser.add_argument("--store", action="store_true",
                        help="do not compress. Much faster, and the runtime is "
                             "mostly DLLs, so the saving is modest.")
    parser.add_argument("--skip-exe-check", action="store_true",
                        help="pack without comparing the EXE against src/")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    version = declared_version()
    numeric = version.split("-")[0]

    problems, docs = check_version_agreement(version)

    if not (DIST / JAR_NAME).is_file():
        problems.append("missing %s -- run build_0.4.2.bat first" % (DIST / JAR_NAME))
    if not (GUI_DIR / "AutoWormGUI.exe").is_file():
        problems.append("missing %s" % (GUI_DIR / "AutoWormGUI.exe"))
    for required in ("models", "licenses", "_internal"):
        if not (GUI_DIR / required).is_dir():
            problems.append("missing %s" % (GUI_DIR / required))

    if problems:
        for line in problems:
            print("ERROR: " + line, file=sys.stderr)
        return 1

    exe = GUI_DIR / "AutoWormGUI.exe"
    if not args.skip_exe_check:
        print()
        print("Checking that the packaged EXE is the current source.")
        print("(Each line compares the code objects inside the EXE with freshly")
        print(" compiled src/*.py: names, literals, numbers, argument layout and")
        print(" the bytecode itself, in the order the compiler emitted them, so a")
        print(" changed number counts as much as a changed message and two")
        print(" swapped constants count as well. Comments and line numbers do")
        print(" not.)")
        matching, differing, missing = report_comparison("new EXE", exe, verbose)
        if differing:
            print(file=sys.stderr)
            print("ERROR: the built EXE does not match src/ for: %s"
                  % ", ".join(differing), file=sys.stderr)
            print("       Rebuild with build_0.4.2.bat before releasing: the zip "
                  "would ship the old code.", file=sys.stderr)
            return 1
        if not matching:
            print("ERROR: no module in the EXE could be compared against src/",
                  file=sys.stderr)
            return 1
        if args.previous_exe:
            print()
            _, old_differing, _ = report_comparison(
                "previous EXE", args.previous_exe, verbose)
            if not old_differing:
                print("ERROR: the previous EXE matches the current src/ too, so "
                      "this comparison cannot show that the rebuild changed "
                      "anything. Pass the EXE of an older build, or drop "
                      "--previous-exe.", file=sys.stderr)
                return 1
            print("ok: the previous EXE differs from src/, so the comparison "
                  "above is not vacuous")

    print()
    sync_docs_to_dist(docs)

    entries = collect_files(docs)
    outdir = Path(args.outdir) if args.outdir else RELEASE_ROOT
    out_zip = outdir / ("AutoWorm-%s-ImageJ.zip" % version)

    print()
    print("Packing %d files into %s" % (len(entries), out_zip))
    if out_zip.exists():
        print("  replacing the existing %s (%.1f MiB)"
              % (out_zip.name, out_zip.stat().st_size / 1048576.0))
    total = write_zip(entries, out_zip, compress=not args.store)

    print()
    print("packed %.2f GiB -> %.2f GiB zip%s"
          % (total / 1073741824.0, out_zip.stat().st_size / 1073741824.0,
             "" if args.store else " (deflate)"))
    print()
    print("SHA256")
    print("  %s\n    %s" % (out_zip.name, sha256_file(out_zip)))
    print("  %s\n    %s" % (JAR_NAME, sha256_file(DIST / JAR_NAME)))
    print("  %s\n    %s" % ("AutoWormGUI.exe", sha256_file(exe)))
    print()
    print("Unpack %s into the Fiji root (the folder holding ImageJ-win64.exe)."
          % out_zip.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
