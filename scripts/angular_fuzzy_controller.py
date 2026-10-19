#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mamdani fuzzy controller for SARA heading regulation."""

import math

import fuzzylite as fl


class AngularFuzzyController:
    """Return angular velocity from heading error and heading-error rate.

    Positive heading error produces positive angular velocity.  The ZERO
    membership is deliberately trapezoidal: full membership between +/-3 deg
    and fading to zero at +/-7 deg.  This complements the planner hysteresis.
    """

    def __init__(self, max_angular_velocity=0.45, max_error_rate=1.5):
        self.max_angular_velocity = float(max_angular_velocity)
        self.max_error_rate = float(max_error_rate)

        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be > 0")
        if self.max_error_rate <= 0.0:
            raise ValueError("max_error_rate must be > 0")

        deg = math.pi / 180.0
        e3 = 3.0 * deg
        e7 = 7.0 * deg
        e25 = 25.0 * deg
        e60 = 60.0 * deg

        w = self.max_angular_velocity
        w_slow = 0.45 * w

        self.engine = fl.Engine(
            name="AngularFuzzyController",
            input_variables=[
                fl.InputVariable(
                    name="heading_error",
                    minimum=-math.pi,
                    maximum=math.pi,
                    lock_range=True,
                    terms=[
                        fl.Trapezoid("NB", -math.pi, -math.pi, -e60, -e25),
                        fl.Triangle("NS", -e60, -e25, -e3),
                        fl.Trapezoid("ZERO", -e7, -e3, e3, e7),
                        fl.Triangle("PS", e3, e25, e60),
                        fl.Trapezoid("PB", e25, e60, math.pi, math.pi),
                    ],
                ),
                fl.InputVariable(
                    name="error_rate",
                    minimum=-self.max_error_rate,
                    maximum=self.max_error_rate,
                    lock_range=True,
                    terms=[
                        fl.Trapezoid(
                            "N",
                            -self.max_error_rate,
                            -self.max_error_rate,
                            -0.30,
                            -0.04,
                        ),
                        fl.Triangle("ZERO", -0.12, 0.0, 0.12),
                        fl.Trapezoid(
                            "P",
                            0.04,
                            0.30,
                            self.max_error_rate,
                            self.max_error_rate,
                        ),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="angular_velocity",
                    minimum=-w,
                    maximum=w,
                    lock_range=True,
                    lock_previous=False,
                    default_value=0.0,
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(resolution=100),
                    terms=[
                        fl.Trapezoid("NB", -w, -w, -0.95 * w, -w_slow),
                        fl.Triangle("NS", -0.75 * w, -w_slow, -0.06 * w),
                        fl.Trapezoid("ZERO", -0.12 * w, -0.02 * w, 0.02 * w, 0.12 * w),
                        fl.Triangle("PS", 0.06 * w, w_slow, 0.75 * w),
                        fl.Trapezoid("PB", w_slow, 0.95 * w, w, w),
                    ],
                )
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="mamdani",
                    conjunction=fl.Minimum(),
                    disjunction=fl.Maximum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        # Around the target: do not fight the chair.
                        fl.Rule.create("if heading_error is ZERO then angular_velocity is ZERO"),

                        # Negative error. Positive error_rate means we are moving toward zero.
                        fl.Rule.create("if heading_error is NB and error_rate is N then angular_velocity is NB"),
                        fl.Rule.create("if heading_error is NB and error_rate is ZERO then angular_velocity is NB"),
                        fl.Rule.create("if heading_error is NB and error_rate is P then angular_velocity is NS"),
                        fl.Rule.create("if heading_error is NS and error_rate is N then angular_velocity is NB"),
                        fl.Rule.create("if heading_error is NS and error_rate is ZERO then angular_velocity is NS"),
                        fl.Rule.create("if heading_error is NS and error_rate is P then angular_velocity is ZERO"),

                        # Positive error. Negative error_rate means we are moving toward zero.
                        fl.Rule.create("if heading_error is PS and error_rate is N then angular_velocity is ZERO"),
                        fl.Rule.create("if heading_error is PS and error_rate is ZERO then angular_velocity is PS"),
                        fl.Rule.create("if heading_error is PS and error_rate is P then angular_velocity is PB"),
                        fl.Rule.create("if heading_error is PB and error_rate is N then angular_velocity is PS"),
                        fl.Rule.create("if heading_error is PB and error_rate is ZERO then angular_velocity is PB"),
                        fl.Rule.create("if heading_error is PB and error_rate is P then angular_velocity is PB"),
                    ],
                )
            ],
        )

    def compute(self, heading_error, error_rate=0.0):
        heading_error = max(-math.pi, min(math.pi, float(heading_error)))
        error_rate = max(
            -self.max_error_rate,
            min(self.max_error_rate, float(error_rate)),
        )

        self.engine.input_variable("heading_error").value = heading_error
        self.engine.input_variable("error_rate").value = error_rate
        self.engine.process()

        value = float(self.engine.output_variable("angular_velocity").value)
        if not math.isfinite(value):
            return 0.0
        return max(-self.max_angular_velocity, min(self.max_angular_velocity, value))

    def reset(self):
        self.engine.restart()
