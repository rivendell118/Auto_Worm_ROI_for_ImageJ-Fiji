"""发布闸门：jar 与 plugin_src/ 的一致性检查。

`make_release.py` 会重新编译 plugin_src/*.java，再逐个比对 jar 里的 class。这道闸门
是唯一会去看 jar 的检查——EXE 那条只管 Python，版本号那条以前读的是 plugin_src/
plugins.config，改没重建 jar 都一样，所以「改了 Java、只跑了 PyInstaller 那一半」
能一路走到打包。这里钉住三件事：

1. class 解析器读得对：自己编译一份源码，逐项与 jar 里的比；
2. 源码一改就报出来，且报的是那几个类；
3. 读不动的时候宁可报「没法比」，也不能悄悄当成「一样」。

解析器必须自己写：标准库没有 class 文件解析。它出过三次错（long/double 占两个常量
池槽位、属性长度被读了两次、字段少读一个 u2），每一次的表现都是「解析越界」——所以
下面专门有一组用例盯着这些边界。
"""
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import make_release as mr

JAVAC = shutil.which("javac")
CLASS_NAMES = ("Auto_Worm_ROI.class", "Auto_Worm_Annotations.class")


@unittest.skipIf(JAVAC is None, "no JDK on PATH: the jar check needs javac")
class CompilesAndParsesTests(unittest.TestCase):
    """真源码、真 javac、真 jar：解析器要能读完整。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.staging = Path(cls.tmp.name) / "classes"
        mr.compile_plugin_to(cls.staging)
        cls.compiled = sorted(path.name for path in cls.staging.glob("*.class"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_class_compiles_and_parses(self):
        self.assertTrue(self.compiled, "javac produced no classes")
        for name in self.compiled:
            with self.subTest(name=name):
                surface = mr.class_surface((self.staging / name).read_bytes())
                self.assertTrue(surface)

    def test_the_jar_holds_these_very_classes(self):
        # The checked-in jar is rebuilt as part of the build, so it should be
        # these bytes. If this fails the jar is stale -- which is exactly what
        # the release gate reports, and a rebuild is the fix.
        entries = mr.jar_entries()
        in_jar = sorted(name for name in entries if name.endswith(".class"))
        self.assertEqual(in_jar, self.compiled)
        for name in self.compiled:
            with self.subTest(name=name):
                self.assertEqual(
                    mr.class_surface(entries[name]),
                    mr.class_surface((self.staging / name).read_bytes()))

    def test_parsing_is_deterministic(self):
        data = (self.staging / self.compiled[0]).read_bytes()
        self.assertEqual(mr.class_surface(data), mr.class_surface(data))

    def test_a_changed_string_literal_is_reported_as_a_difference(self):
        name = "Auto_Worm_Annotations.class"
        data = (self.staging / name).read_bytes()
        self.assertIn(b"head_", data)
        self.assertNotEqual(mr.class_surface(data),
                            mr.class_surface(data.replace(b"head_", b"heax_", 1)))

    def test_source_metadata_alone_is_not_a_difference(self):
        # 元数据属性（行号表、局部变量名）在比较时被丢掉：改了不改行为的东西不算
        # 差异，否则每次重排注释都会让闸门误报。
        for attribute in ("LineNumberTable", "LocalVariableTable", "SourceFile"):
            self.assertIn(attribute, mr.JAR_INFO_ATTRS)

    def test_a_truncated_class_is_refused_rather_than_treated_as_equal(self):
        # 读不动就说读不动。要是解析器在这里截断了事，两个不同的 class 会被判成
        # 「一样」，闸门也就白装了。
        data = (self.staging / self.compiled[0]).read_bytes()
        with self.assertRaises(ValueError):
            mr.class_surface(data[:len(data) - 40])

    def test_trailing_bytes_are_refused(self):
        data = (self.staging / self.compiled[0]).read_bytes()
        with self.assertRaises(ValueError):
            mr.class_surface(data + b"\x00\x00")


class StaleJarTests(unittest.TestCase):
    """改了源码、jar 没重建时，必须报出来。"""

    def _swap_jar(self):
        """Point the module at a jar built from a modified copy of the source."""
        return mr.DIST / mr.JAR_NAME

    @unittest.skipIf(JAVAC is None, "no JDK on PATH: the jar check needs javac")
    def test_a_jar_built_from_older_source_is_reported(self):
        jar = self._swap_jar()
        original = jar.read_bytes()
        # Restored here rather than through addCleanup: the point of the check
        # below is that this test put a different jar in place, and a cleanup
        # registered with the framework runs after the assertions, not before.
        try:
            # Rebuild the jar from a source tree with one literal changed, the
            # way a stale jar comes about: the source moved on, the jar did not.
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / "plugin_src"
                shutil.copytree(mr.PLUGIN_SRC, work)
                target = work / "Auto_Worm_Annotations.java"
                target.write_text(
                    target.read_text(encoding="utf-8").replace(
                        '"head_"', '"heax_"', 1),
                    encoding="utf-8")
                saved, mr.PLUGIN_SRC = mr.PLUGIN_SRC, work
                try:
                    staging = Path(tmp) / "classes"
                    mr.compile_plugin_to(staging)
                    with zipfile.ZipFile(jar, "w") as archive:
                        for path in sorted(staging.glob("*.class")):
                            archive.write(path, path.name)
                        archive.write(mr.PLUGIN_CONFIG, "plugins.config")
                finally:
                    mr.PLUGIN_SRC = saved

            problems, notes = mr.compare_jar_to_source()
            self.assertTrue(problems, "a jar built from older source was accepted")
            self.assertTrue(any("Auto_Worm_Annotations" in p for p in problems),
                            "the stale class was not named: %r" % (problems,))
            self.assertIn("Auto_Worm_ROI.class", notes,
                          "the untouched class should still compare equal")
        finally:
            jar.write_bytes(original)

    def test_a_jar_without_plugins_config_is_reported(self):
        original = self._swap_jar().read_bytes()
        self.addCleanup(self._swap_jar().write_bytes, original)
        entries = mr.jar_entries()
        with zipfile.ZipFile(self._swap_jar(), "w") as archive:
            for name, data in entries.items():
                if name != "plugins.config":
                    archive.writestr(name, data)
        problems, _ = mr.compare_jar_to_source()
        self.assertTrue(any("plugins.config" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
