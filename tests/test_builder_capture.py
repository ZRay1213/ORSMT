import unittest
from orsmt.capture import capture_builder

class CaptureTests(unittest.TestCase):
    def test_native_builder(self):
        m,details=capture_builder('''
import pulp
def build_model():
    m=pulp.LpProblem('captured'); x=pulp.LpVariable('x',0,5,pulp.LpInteger)
    m += x >= 2
    m += x+3
    return m
''')
        self.assertEqual(m.objective.expression.constant,3)
        self.assertEqual(m.variable_map['x'].kind,'integer')
        self.assertEqual(details['native_translation']['format'],'pulp')
    def test_missing_protocol_and_timeout(self):
        with self.assertRaisesRegex(ValueError,'build_model'):capture_builder('a=1')
        with self.assertRaises(TimeoutError):capture_builder('while True: pass',timeout=0.2)

if __name__=='__main__':unittest.main()
