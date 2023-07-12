#!/usr/bin/env python3
'''Test Package'''

import bazelpybind11.python.pyfoo as pf
from bazelpybind11.python.pyfoo import Foo

print(f'python path: {sys.path}')

print(f'version: {bazelpybind11.__version__}')

# foo
print(f'pf.Foo: {dir(pf.Foo)}')

pf.free_function(2147483647) # max int
pf.free_function(2147483647+1) # max int + 1

f = pf.Foo()
print(f'class Foo: {dir(f)}')
f.static_function(1)
f.static_function(2147483647)
f.static_function(2147483647+1)
f.int = 13
assert(f.int == 13)
f.int64 = 31
assert(f.int64 == 31)
