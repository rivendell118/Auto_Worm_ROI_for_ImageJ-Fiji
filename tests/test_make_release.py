"""The release fingerprint: it must catch a changed number, and must not invent
differences between src/ and the copy of a module inside the EXE.

The check exists to stop a zip that ships a GUI built from older source. It
compared names and string literals only, so `x + 1` -> `x - 99` produced a
fingerprint identical to the original and the check passed on a stale EXE. These
tests pin both halves: the edit is caught, and an unchanged module fingerprints
the same whether it was compiled from source or read back out of a .pyc the way
PyInstaller stores it.

A second review found the repair was still blind to reordering. Collecting the
lines into a set throws away which constant sits at which index and which code
object owns which name, so `a = 1; b = 2` and `a = 2; b = 1` -- same bytecode,
swapped constant table -- compared equal. Those cases are pinned below as well,
and the group of tests that follows them holds the set-shaped ones to what they
were meant to fix.
"""

import marshal
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_release import code_surface, constant_key


def surface(source):
    """The fingerprint of one module of source."""
    return code_surface(compile(source, "<test>", "exec"), [])


class CodeSurfaceDifferenceTests(unittest.TestCase):
    """Edits that change what the program does must change the fingerprint."""

    def test_a_changed_number_is_a_difference(self):
        # The review's example: no name and no string changes, so the old
        # fingerprint could not see this at all.
        self.assertNotEqual(surface("def f(x):\n    return x + 1\n"),
                            surface("def f(x):\n    return x - 99\n"))

    def test_a_changed_number_in_a_module_constant_is_a_difference(self):
        self.assertNotEqual(surface("LIMIT = 580\n"),
                            surface("LIMIT = 579\n"))

    def test_a_changed_string_is_still_a_difference(self):
        self.assertNotEqual(surface('MESSAGE = "before"\n'),
                            surface('MESSAGE = "after"\n'))

    def test_a_changed_looked_up_name_is_still_a_difference(self):
        self.assertNotEqual(surface("def f(x):\n    return g(x)\n"),
                            surface("def f(x):\n    return h(x)\n"))

    def test_a_change_inside_a_nested_function_is_a_difference(self):
        self.assertNotEqual(
            surface("def outer():\n    def inner():\n        return 1\n"
                    "    return inner\n"),
            surface("def outer():\n    def inner():\n        return 2\n"
                    "    return inner\n"))

    def test_a_changed_argument_list_is_a_difference(self):
        self.assertNotEqual(surface("def f(x):\n    return x\n"),
                            surface("def f(x, y):\n    return x\n"))

    def test_a_changed_default_argument_is_a_difference(self):
        # A default is a constant of the enclosing code object, not of the
        # function itself, so it can be missed by a per-function comparison.
        self.assertNotEqual(surface("def f(x=1):\n    return x\n"),
                            surface("def f(x=2):\n    return x\n"))

    def test_two_swapped_module_constants_are_a_difference(self):
        # The review's example. Both compile to the same instructions -- the
        # bytecode loads constant 1 and then constant 2 either way -- and only
        # the constant table moved, so a row of the fingerprint that ignores the
        # position of a constant cannot see it. The assertion on the bytecode is
        # the point of the test, not decoration: it is why the other rows have
        # to carry the order.
        before, after = "a = 1\nb = 2\n", "a = 2\nb = 1\n"
        self.assertEqual(compile(before, "<test>", "exec").co_code,
                         compile(after, "<test>", "exec").co_code)
        self.assertNotEqual(surface(before), surface(after))

    def test_two_functions_exchanging_bodies_are_a_difference(self):
        self.assertNotEqual(surface("def f():\n    return 1\n\n"
                                    "def g():\n    return 2\n"),
                            surface("def f():\n    return 2\n\n"
                                    "def g():\n    return 1\n"))

    def test_a_nested_function_moving_position_is_a_difference(self):
        # Same constants, same names, same instructions; the function body and
        # the number simply changed places in co_consts, which means one of them
        # now belongs to a different code object.
        self.assertNotEqual(surface("def f():\n    return 1\n\nX = 2\n"),
                            surface("X = 2\n\ndef f():\n    return 1\n"))

    def test_swapped_values_in_a_dict_literal_are_a_difference(self):
        self.assertNotEqual(surface('M = {"x": 1, "y": 2}\n'),
                            surface('M = {"x": 2, "y": 1}\n'))


class CodeSurfaceStabilityTests(unittest.TestCase):
    """Unchanged code must fingerprint the same, whichever way it is reached."""

    def test_a_change_a_compiler_would_undo_is_not_a_difference(self):
        # Line numbers are left out deliberately: only whitespace and comments
        # moved, which cannot change what the program does.
        self.assertEqual(surface("def f(x):\n    return x + 1\n"),
                         surface("\n\n# a note\ndef f(x):\n    return x + 1\n"))

    def test_reading_a_module_back_out_of_a_pyc_is_not_a_difference(self):
        # What the EXE holds is a marshalled code object, so this is the
        # comparison the release check actually makes, for every module.
        modules = sorted((ROOT / "src").glob("*.py"))
        self.assertTrue(modules, "no src/*.py to check")
        for path in modules:
            with self.subTest(module=path.name):
                compiled = compile(path.read_bytes(), str(path), "exec")
                reloaded = marshal.loads(marshal.dumps(compiled))
                self.assertEqual(code_surface(compiled, []),
                                 code_surface(reloaded, []))

    def test_the_order_a_reordering_is_judged_by_survives_a_pyc_round_trip(self):
        # The release check never sees freshly compiled source on both sides: one
        # side comes back out of the EXE. Order has to survive marshalling for
        # the reordering tests above to mean anything in the check itself.
        def through_a_pyc(source):
            return code_surface(
                marshal.loads(marshal.dumps(compile(source, "<test>", "exec"))), [])

        self.assertTrue(surface("a = 1\nb = 2\n") == through_a_pyc("a = 1\nb = 2\n"))
        self.assertNotEqual(through_a_pyc("a = 1\nb = 2\n"),
                            through_a_pyc("a = 2\nb = 1\n"))

    def test_an_unordered_container_renders_the_same_from_any_insertion_order(self):
        # repr() of a frozenset follows its internal table, which is filled from
        # source order when compiling and from stored order when unmarshalled --
        # so an unchanged frozenset really does print two ways. Without
        # normalising it, every module carrying one is reported as changed.
        elements = ["hd95", "assd", "endpoint_error_fraction",
                    "area_error_fraction"]
        self.assertEqual(constant_key(frozenset(elements)),
                         constant_key(frozenset(reversed(elements))))

    def test_an_unordered_container_still_reports_its_contents(self):
        self.assertNotEqual(constant_key(frozenset(["a"])),
                            constant_key(frozenset(["b"])))
        # A set and a frozenset are not the same constant.
        self.assertNotEqual(constant_key(frozenset(["a"])),
                            constant_key(set(["a"])))
        # A tuple is ordered, so a reordering is a real difference.
        self.assertNotEqual(constant_key(("a", "b")), constant_key(("b", "a")))

    def test_a_container_holding_a_container_is_handled(self):
        self.assertEqual(constant_key((frozenset(["a", "b"]), "c")),
                         constant_key((frozenset(["b", "a"]), "c")))
        self.assertNotEqual(constant_key((frozenset(["a", "b"]), "c")),
                            constant_key((frozenset(["a", "b"]), "d")))


if __name__ == "__main__":
    unittest.main()
