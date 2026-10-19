#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mamdani fuzzy controller for SARA forward velocity."""

import math

import fuzzylite as fl


class LinearFuzzyController:
    """Return a positive linear velocity from waypoint distance.

    This is intentionally a small nonlinear speed scheduler.  The planner is
    responsible for heading alignment, waypoint acceptance, and command-rate
    limiting.
    """

    def __init__(self, max_linear_velocity=0.25, distance_universe=4.0):
        self.max_linear_velocity = float(max_linear_velocity)
        self.distance_universe = float(distance_universe)

        if self.max_linear_velocity <= 0.0:
            raise ValueError("max_linear_velocity must be > 0")
        if self.distance_universe <= 0.0:
            raise ValueError("distance_universe must be > 0")

        v = self.max_linear_velocity

        self.engine = fl.Engine(
            name="LinearFuzzyController",
            input_variables=[
                fl.InputVariable(
                    name="distance",
                    minimum=0.0,
                    maximum=self.distance_universe,
                    lock_range=True,
                    terms=[
                        fl.Trapezoid("ZERO", 0.0, 0.0, 0.05, 0.12),
                        fl.Triangle("NEAR", 0.05, 0.30, 0.80),
                        fl.Triangle("MEDIUM", 0.45, 1.20, 2.20),
                        fl.Trapezoid(
                            "FAR",
                            1.50,
                            2.50,
                            self.distance_universe,
                            self.distance_universe,
                        ),
                    ],
                )
            ],
            output_variables=[
                fl.OutputVariable(
                    name="linear_velocity",
                    minimum=0.0,
                    maximum=v,
                    lock_range=True,
                    lock_previous=False,
                    default_value=0.0,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(resolution=100),
                    terms=[
                        fl.Trapezoid("ZERO", 0.0, 0.0, 0.015 * v, 0.12 * v),
                        fl.Triangle("SLOW", 0.08 * v, 0.38 * v, 0.65 * v),
                        fl.Triangle("MEDIUM", 0.45 * v, 0.68 * v, 0.88 * v),
                        fl.Trapezoid("FAST", 0.72 * v, 0.92 * v, v, v),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="mamdani",
                    conjunction=None,
                    disjunction=None,
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if distance is ZERO then linear_velocity is ZERO"),
                        fl.Rule.create("if distance is NEAR then linear_velocity is SLOW"),
                        fl.Rule.create("if distance is MEDIUM then linear_velocity is MEDIUM"),
                        fl.Rule.create("if distance is FAR then linear_velocity is FAST"),
                    ],
                )
            ],
        )

    def compute(self, distance):
        distance = max(0.0, min(self.distance_universe, float(distance)))
        self.engine.input_variable("distance").value = distance
        self.engine.process()

        value = float(self.engine.output_variable("linear_velocity").value)
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(self.max_linear_velocity, value))

    def reset(self):
        self.engine.restart()
