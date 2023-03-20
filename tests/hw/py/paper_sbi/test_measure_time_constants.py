#!/usr/bin/env python

import unittest

from paper_sbi.scripts.measure_tau_mem import measure_tau_mem
from paper_sbi.scripts.measure_tau_icc import measure_tau_icc


class TestTimeConstants(unittest.TestCase):
    '''
    Test measurement of membrane and inter-compartment time constants.
    '''
    parameters = [100, 1000, 2]

    def test_measure_tau_mem(self):
        measure_tau_mem(self.parameters)

    def test_measure_tau_icc(self):
        measure_tau_icc(self.parameters)


if __name__ == "__main__":
    unittest.main()
