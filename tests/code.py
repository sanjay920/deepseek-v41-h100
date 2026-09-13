"""Independent checks; execute inside an isolated container with /candidate.py."""
import importlib.util
import json
import random
import unittest

spec=importlib.util.spec_from_file_location('candidate','/candidate.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
rng=random.Random(912)
for _ in range(2000):
    text=''.join(rng.choices('abcABC 012é界🙂',k=rng.randrange(0,100)))
    expected=next((c for c in text if text.count(c)==1),None)
    assert module.first_unique_char(text)==expected
print(json.dumps({'passed':result.wasSuccessful(),'generated_tests':result.testsRun,'independent_cases':2000}))
assert result.wasSuccessful()
