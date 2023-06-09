#!/usr/bin/env python

import unittest

import numpy as np

from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.scripts.record_variations import main as \
    record_variations

from paper_sbi.scripts.attenuation_abc import main as abc

from paramopt.abc import Algorithm

class TestABCBSS(unittest.TestCase):
    '''
    Test the SNPE algorithm on BSS-2 and :class:`Observation.LENGTH_CONSTANT`
    as an observation.


    All other :class:`Algorithm` and :class:`Observation` are tested in
    software.
    '''

    observation = Observation.LENGTH_CONSTANT

    @classmethod
    def setUpClass(cls):
        cls.target_df = record_variations(length=4, repetitions=20)

    def test_snpe(self):
        simulations = [50, 50]
        samples, posteriors = abc(self.target_df,
                                  self.observation,
                                  Algorithm.SNPE,
                                  simulations=simulations,
                                  global_parameters=True)

        self.assertEqual(len(samples), np.sum(simulations))
        self.assertEqual(len(posteriors), len(simulations))


if __name__ == "__main__":
    unittest.main()
