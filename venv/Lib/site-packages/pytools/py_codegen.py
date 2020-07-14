from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import marshal
from types import FunctionType, ModuleType

import six


try:
    from importlib.util import MAGIC_NUMBER as BYTECODE_VERSION
except ImportError:
    # Pre-3.4
    import imp
    BYTECODE_VERSION = imp.get_magic()


# loosely based on
# http://effbot.org/zone/python-code-generator.htm

class Indentation(object):
    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        self.generator.indent()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.dedent()


class PythonCodeGenerator(object):
    def __init__(self):
        self.preamble = []
        self.code = []
        self.level = 0

    def extend(self, sub_generator):
        for line in sub_generator.code:
            self.code.append(" "*(4*self.level) + line)

    def get(self):
        result = "\n".join(self.code)
        if self.preamble:
            result = "\n".join(self.preamble) + "\n" + result
        return result

    def add_to_preamble(self, s):
        self.preamble.append(s)

    def __call__(self, s):
        if not s.strip():
            self.code.append("")
        else:
            if "\n" in s:
                s = remove_common_indentation(s)

            for line in s.split("\n"):
                self.code.append(" "*(4*self.level) + line)

    def indent(self):
        self.level += 1

    def dedent(self):
        if self.level == 0:
            raise RuntimeError("internal error in python code generator")
        self.level -= 1

    def get_module(self, name="<generated code>"):
        result_dict = {}
        source_text = self.get()
        exec(compile(
            source_text.rstrip()+"\n", name, "exec"),
                result_dict)
        result_dict["_MODULE_SOURCE_CODE"] = source_text
        return result_dict

    def get_picklable_module(self):
        return PicklableModule(self.get_module())


class PythonFunctionGenerator(PythonCodeGenerator):
    def __init__(self, name, args):
        PythonCodeGenerator.__init__(self)
        self.name = name

        self("def %s(%s):" % (name, ", ".join(args)))
        self.indent()

    def get_function(self):
        return self.get_module()[self.name]

    def get_picklable_function(self):
        module = self.get_picklable_module()
        return PicklableFunction(module, self.name)


# {{{ pickling of binaries for generated code

def _get_empty_module_dict():
    result_dict = {}
    exec(compile("", "<generated function>", "exec"), result_dict)
    return result_dict


_empty_module_dict = _get_empty_module_dict()


class PicklableModule(object):
    def __init__(self, mod_globals):
        self.mod_globals = mod_globals

    def __getstate__(self):
        nondefault_globals = {}
        functions = {}
        modules = {}

        for k, v in six.iteritems(self.mod_globals):
            if isinstance(v, FunctionType):
                functions[k] = (
                        v.__name__,
                        marshal.dumps(v.__code__),
                        v.__defaults__)
            elif isinstance(v, ModuleType):
                modules[k] = v.__name__
            elif k not in _empty_module_dict:
                nondefault_globals[k] = v

        return (1, BYTECODE_VERSION, functions, modules, nondefault_globals)

    def __setstate__(self, obj):
        if obj[0] == 0:
            magic, functions, nondefault_globals = obj[1:]
            modules = {}
        elif obj[0] == 1:
            magic, functions, modules, nondefault_globals = obj[1:]
        else:
            raise ValueError("unknown version of PicklableModule")

        if magic != BYTECODE_VERSION:
            raise ValueError("cannot unpickle function binary: "
                    "incorrect magic value (got: %r, expected: %r)"
                    % (magic, BYTECODE_VERSION))

        mod_globals = _empty_module_dict.copy()
        mod_globals.update(nondefault_globals)

        from pytools.importlib_backport import import_module
        for k, mod_name in six.iteritems(modules):
            mod_globals[k] = import_module(mod_name)

        for k, (name, code_bytes, argdefs) in six.iteritems(functions):
            f = FunctionType(
                    marshal.loads(code_bytes), mod_globals, name=name,
                    argdefs=argdefs)
            mod_globals[k] = f

        self.mod_globals = mod_globals

# }}}


# {{{ picklable function

class PicklableFunction(object):
    """Convience class wrapping a function in a :class:`PicklableModule`.
    """

    def __init__(self, module, name):
        self._initialize(module, name)

    def _initialize(self, module, name):
        self.module = module
        self.name = name
        self.func = module.mod_globals[name]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getstate__(self):
        return {"module": self.module, "name": self.name}

    def __setstate__(self, obj):
        self._initialize(obj["module"], obj["name"])

# }}}


# {{{ remove common indentation

def remove_common_indentation(code, require_leading_newline=True):
    if "\n" not in code:
        return code

    if require_leading_newline and not code.startswith("\n"):
        return code

    lines = code.split("\n")
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation")

    return "\n".join(line[base_indent:] for line in lines)

# }}}

# vim: foldmethod=marker
