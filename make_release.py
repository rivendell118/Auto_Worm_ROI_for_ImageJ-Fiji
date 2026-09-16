"""Assemble the distributable zip from an existing build under dist/.

ImageJ installs a plug-in by its position in the Fiji tree, so the zip is laid
out the way Fiji expects to find things -- plugins/ at the top -- and is meant
to be unpacked into the Fiji root, where ImageJ-win64.exe is:

    plugins/Auto_Worm_ROI.jar
    plugins/AutoWormImageJ/AutoWormGUI.exe
    plugins/AutoWormImageJ/{models,_internal,licenses}/...
    README_0.4.2_ImageJ_Java8.md  CHANGELOG_0.4.2.md  VALIDATION_0.4.2.md  LICENSE

Four checks run before anything is packed, because each one covers a mistake
that produces a zip which looks completely normal:

  * the version string agrees in every place that carries it, so a partial bump
    cannot ship a jar that announces one version and a summary CSV that records
    another;
  * every module the EXE shares with src/ is the same code as src/ -- compares
    the packaged code objects against freshly compiled ones, module by module,
    rather than searching for a marker string somebody has to remember to add;
  * the jar holds the current Java source, compiled the way the build compiles
    it, and its embedded plugins.config is the current one;
  * when --previous-exe is given, that older build must NOT match the current
    source. Without that half, a matching comparison proves nothing about
    whether the rebuild did anything at all.

The jar check is separate from the EXE check because nothing else looks at the
jar. Editing plugin_src/*.java and running only the PyInstaller half of the
build is an ordinary way to work -- the EXE check passes, every Python test
passes, and the zip ships a jar built from Java that no longer exists. The
version check used to read plugin_src/plugins.config, which is the file that
would be edited by hand and is present whether or not the jar was rebuilt, so
it could not catch that either; it now reads the copy inside the jar.

Run it with the build interpreter, after build_0.4.2.bat:

    <build python> make_release.py [--previous-exe <old AutoWormGUI.exe>]
"""

from __future__ import print_function

import argparse
import hashlib
import marshal
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
GUI_DIR = DIST / "AutoWormImageJ"
SRC = ROOT / "src"
PLUGIN_SRC = ROOT / "plugin_src"
PLUGIN_CONFIG = PLUGIN_SRC / "plugins.config"
IJ_JAR = ROOT / "lib" / "ij.jar"
RELEASE_ROOT = ROOT / "release"

# Exactly the arguments build_0.4.2.bat compiles the plug-in with. They are
# repeated here rather than shared, because the comparison is only meaningful
# when the release script recompiles the same way the build did -- a different
# --release target changes the emitted bytecode and the check would then report
# a difference that is the script's own doing. If the build script's javac line
# changes, this has to change with it, and both halves are in one place to see.
JAVAC_FLAGS = ("--release", "8", "-encoding", "UTF-8")

# Class-file attributes that cannot change what the plug-in does: they describe
# where the code came from or help a verifier, and two compilations of the same
# source may differ in them without the program differing at all. Dropped so the
# comparison is over structure and instructions only.
JAR_INFO_ATTRS = frozenset((
    "SourceFile", "LineNumberTable", "LocalVariableTable",
    "LocalVariableTypeTable", "StackMapTable", "StackMap",
    "RuntimeVisibleTypeAnnotations", "RuntimeInvisibleTypeAnnotations",
    "MethodParameters",
))

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

    Read out of the jar, which is what ships and what ImageJ actually parses.
    Reading plugin_src/plugins.config instead answers a different question -- it
    says what the version would be if the jar were rebuilt -- and the answer is
    the same whether or not it was, so a stale jar carrying an old menu string
    sailed through. Only when the jar is not built yet does this fall back to
    the source file, where the version is at least what the next build will
    carry, and that case is reported as a missing artifact before anything is
    packed.
    """
    if (DIST / JAR_NAME).is_file():
        with zipfile.ZipFile(DIST / JAR_NAME) as archive:
            try:
                text = archive.read("plugins.config").decode("ascii")
            except KeyError:
                raise SystemExit("%s has no plugins.config inside it -- it is "
                                 "not the plug-in jar" % (DIST / JAR_NAME))
        origin = "%s!plugins.config" % JAR_NAME
    else:
        text = PLUGIN_CONFIG.read_text(encoding="ascii")
        origin = str(PLUGIN_CONFIG)
    match = VERSION_PATTERN.search(text)
    if not match:
        raise SystemExit("no version found in %s" % origin)
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


class ClassReader(object):
    """Just enough class-file parsing to walk one, attribute by attribute.

    A class file is a flat sequence of length-prefixed structures, so reading
    past the part being compared has to be exact -- one wrong length and every
    later field is read out of the middle of something else. That is the whole
    reason this exists instead of a regex over the bytes.

    Anything not understood raises, and callers turn that into "cannot compare"
    rather than "matches": a parser that quietly skipped a structure it did not
    recognise would be able to call two different classes equal, which is the
    one failure this check must not have.
    """

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u1(self):
        value = self.data[self.pos]
        self.pos += 1
        return value

    def u2(self):
        value = struct.unpack_from(">H", self.data, self.pos)[0]
        self.pos += 2
        return value

    def u4(self):
        value = struct.unpack_from(">I", self.data, self.pos)[0]
        self.pos += 4
        return value

    def raw(self, count):
        if self.pos + count > len(self.data):
            raise ValueError("class file ends in the middle of a structure")
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def skip_attributes(self):
        for _ in range(self.u2()):
            self.u2()                 # name index
            self.raw(self.u4())       # length + body


def class_surface(data):
    """A comparable rendering of a class file, without the source metadata.

    Everything that decides what the class does is kept: the constant pool, the
    access flags, the interfaces, the fields, and every method with its
    descriptor, its flags and its bytecode. Everything that only records where
    the code came from (see JAR_INFO_ATTRS) is dropped, so a comment, a blank
    line or a local variable name does not count as a difference while a changed
    string literal, a changed number or a changed branch does.
    """
    reader = ClassReader(data)
    if reader.u4() != 0xCAFEBABE:
        raise ValueError("not a class file")
    out = []
    reader.u2()                       # minor version
    reader.u2()                       # major version -- the --release target

    pool = []
    # Entries are numbered from 1, and a long or a double occupies two numbers
    # -- the second is unusable and has no bytes of its own. Pool indices
    # elsewhere count those two slots, so the extra entry is appended to keep
    # every later index lining up with the number the class file used.
    remaining = reader.u2() - 1
    while remaining > 0:
        tag = reader.u1()
        remaining -= 1
        if tag == 1:
            length = reader.u2()
            # Decoded so a changed literal reads as a changed literal rather
            # than as a changed length. Modified UTF-8 only differs from UTF-8
            # for characters that cannot appear in this source.
            pool.append(reader.raw(length).decode("utf-8", "replace"))
        elif tag in (3, 4):
            pool.append(reader.raw(4).hex())
        elif tag in (5, 6):
            pool.append(reader.raw(8).hex())
            pool.append("<two slots>")
            remaining -= 1
        elif tag in (7, 8, 16, 19, 20):
            pool.append(reader.u2())
        elif tag in (9, 10, 11, 12, 17, 18):
            pool.append((reader.u2(), reader.u2()))
        elif tag == 15:
            pool.append((reader.u1(), reader.u2()))
        else:
            raise ValueError("unknown constant pool tag %d" % tag)
    out.append("pool " + repr(pool))

    reader.u2()                       # access flags
    reader.u2()                       # this class
    reader.u2()                       # super class
    out.append("interfaces " + repr([reader.u2() for _ in range(reader.u2())]))
    # Fields carry no bytecode, so names and descriptors are the whole of them.
    # Four reads per field: access flags, name index, descriptor index, and the
    # count of attributes -- which is zero for a field and still has to be
    # consumed. Reading one fewer leaves two bytes unconsumed per field, and the
    # reader then walks into the middle of the next field.
    for _ in range(reader.u2()):
        reader.u2()                   # access flags
        name = pool[reader.u2() - 1]
        descriptor = pool[reader.u2() - 1]
        out.append("field %r %r" % (name, descriptor))
        reader.skip_attributes()
    for _ in range(reader.u2()):
        reader.u2()                   # access flags
        name = pool[reader.u2() - 1]
        descriptor = pool[reader.u2() - 1]
        out.append("method %r %r" % (name, descriptor))
        for _ in range(reader.u2()):
            attribute = pool[reader.u2() - 1]
            body_length = reader.u4()
            if attribute == "Code":
                body = ClassReader(reader.raw(body_length))
                body.u2()             # max stack
                body.u2()             # max locals
                out.append("  bytecode " + body.raw(body.u4()).hex())
                # Exception handlers are branches, so they are code.
                out.append("  handlers " + repr(
                    [(body.u2(), body.u2(), body.u2(), body.u2() or None)
                     for _ in range(body.u2())]))
                body.skip_attributes()
                if body.pos != len(body.data):
                    raise ValueError("trailing bytes in Code")
            elif attribute in JAR_INFO_ATTRS:
                reader.raw(body_length)
            else:
                out.append("  %s %s" % (attribute, reader.raw(body_length).hex()))
    for _ in range(reader.u2()):
        attribute = pool[reader.u2() - 1]
        body_length = reader.u4()
        if attribute in JAR_INFO_ATTRS:
            reader.raw(body_length)
        else:
            out.append("class %s %s" % (attribute, reader.raw(body_length).hex()))
    if reader.pos != len(data):
        raise ValueError("trailing bytes after the class")
    return out


def jar_entries():
    """{name inside the jar: raw bytes} for everything the jar carries."""
    with zipfile.ZipFile(DIST / JAR_NAME) as archive:
        return {info.filename: archive.read(info.filename)
                for info in archive.infolist() if not info.is_dir()}


def compile_plugin_to(destination):
    """Compile plugin_src/*.java the way the build compiles it."""
    units = sorted(str(path) for path in PLUGIN_SRC.glob("*.java"))
    if not units:
        raise SystemExit("no Java sources under %s" % PLUGIN_SRC)
    command = ["javac"] + list(JAVAC_FLAGS) + [
        "-cp", str(IJ_JAR), "-d", str(destination)] + units
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit("javac failed while checking the jar:\n%s%s"
                         % (completed.stdout, completed.stderr))
    return sorted(path.name for path in destination.glob("*.class"))


def compare_jar_to_source():
    """Whether the jar is the current Java source, compiled as the build does.

    Returns (problems, notes). Compiles into a throwaway directory rather than
    build/plugin_classes, so running the release check does not touch the build
    tree it is inspecting.
    """
    problems, notes = [], []
    entries = jar_entries()

    embedded = entries.get("plugins.config")
    if embedded is None:
        problems.append("the jar has no plugins.config in it")
    elif embedded != PLUGIN_CONFIG.read_bytes():
        # Compared as bytes: this file only goes in by copy, so any difference
        # at all means the jar is not carrying the menu the source declares.
        problems.append("plugins.config inside the jar differs from "
                        "plugin_src/plugins.config")

    with tempfile.TemporaryDirectory() as staging:
        compiled = compile_plugin_to(Path(staging))
        in_jar = sorted(name for name in entries if name.endswith(".class"))
        for name in sorted(set(in_jar) - set(compiled)):
            problems.append("%s is in the jar but no longer compiled from "
                            "plugin_src/" % name)
        for name in sorted(set(compiled) - set(in_jar)):
            problems.append("%s compiles from plugin_src/ but is not in the "
                            "jar" % name)
        for name in sorted(set(compiled) & set(in_jar)):
            try:
                fresh = class_surface((Path(staging) / name).read_bytes())
                shipped = class_surface(entries[name])
            except ValueError as error:
                problems.append("%s could not be compared (%s)" % (name, error))
                continue
            if fresh == shipped:
                notes.append(name)
                continue
            problems.append("%s was compiled from different source than the "
                            "jar holds" % name)
            # The first line that differs, for a report that says what changed
            # rather than only that something did.
            for index, (a, b) in enumerate(zip(fresh, shipped)):
                if a != b:
                    problems.append("    first difference at %s piece %d: "
                                    "source %r, jar %r"
                                    % (name, index, a[:120], b[:120]))
                    break
            else:
                problems.append("    the two have different lengths: %d vs %d "
                                "pieces" % (len(fresh), len(shipped)))
    return problems, notes


def report_jar_comparison(verbose):
    problems, notes = compare_jar_to_source()
    print("Checking that the jar is the current Java source.")
    print("(plugin_src/*.java is recompiled the way build_0.4.2.bat compiles")
    print(" it, and every class is compared with the one in the jar: constant")
    print(" pool, methods and bytecode. Line numbers, local variable names and")
    print(" other source metadata do not count.)")
    print("jar: %d class(es) match plugin_src/" % len(notes))
    if verbose and notes:
        print("    match: %s" % ", ".join(notes))
    for line in problems:
        print(line, file=sys.stderr)
    return problems


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
    parser.add_argument("--skip-jar-check", action="store_true",
                        help="pack without comparing the jar against "
                             "plugin_src/ (needs a JDK on PATH)")
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

    if not args.skip_jar_check:
        if shutil.which("javac") is None:
            # Not skipped quietly: the check is the only thing that looks at the
            # jar, and the JDK is on the build machine that produces it.
            print("ERROR: javac is not on PATH, so the jar cannot be compared "
                  "with plugin_src/. Run this where the build runs, or pass "
                  "--skip-jar-check to pack anyway.", file=sys.stderr)
            return 1
        print()
        jar_problems = report_jar_comparison(verbose)
        if jar_problems:
            print(file=sys.stderr)
            print("ERROR: the jar does not match plugin_src/. Rebuild with "
                  "build_0.4.2.bat before releasing: the zip would ship Java "
                  "code that is no longer in the sources.", file=sys.stderr)
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
