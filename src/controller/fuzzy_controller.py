#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import fuzzylite as fl


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FuzzyController:
    def __init__(self, min_linear_speed = 0.25, max_linear_speed = 0.65, min_angular_speed = 0.85, max_angular_speed = 1.00, goal_tolerance = 0.05):
        # ==============================================================
        # TUNING -- edit here, not in the launch file
        # ==============================================================
        self.goal_tolerance = goal_tolerance
        self.distance_max = 4.0

        # Explicit physical output ranges.
        self.linear_min = -1.0
        self.linear_max = 1.0
        self.angular_min = -2.0
        self.angular_max = 2.0

        self.engine = fl.Engine(
            name="SARAWaypointController",
            description="Unified fuzzy waypoint controller for SARA",
            input_variables=[
                fl.InputVariable(
                    name="distanceError",
                    description="Euclidean distance to active waypoint [m]",
                    enabled=True,
                    minimum=-math.inf,
                    maximum=math.inf,
                    lock_range=False,
                    terms=[
                        fl.Triangle("NVeryFar", -math.inf, -0.75, -0.45),
                        fl.Triangle("NFar", -0.75, -0.45, -0.150),
                        fl.Triangle("NNear", -0.45, -0.150, -0.050),
                        fl.Trapezoid("Zero", -0.150, -0.050, 0.050, 0.150),
                        fl.Triangle("PNear", 0.05, 0.15, 0.45),
                        fl.Triangle("PFar", 0.15, 0.45, 0.75),
                        fl.Triangle("PVeryFar", 0.45, 0.75, math.inf),
                    ],
                ),
                fl.InputVariable(
                    name="angleError",
                    description="Wrapped heading error to active waypoint [rad]",
                    enabled=True,
                    minimum=-math.pi,
                    maximum=math.pi,
                    lock_range=False,
                    terms=[
                        fl.Triangle("NVeryFar", -math.pi, -0.044, -0.035),
                        fl.Triangle("NFar", -0.044, -0.035, -0.026),
                        fl.Triangle("NNear", -0.035, -0.026, -0.017),
                        fl.Trapezoid("Zero", -0.026, -0.017, 0.017, 0.026),
                        fl.Triangle("PNear", 0.017, 0.026, 0.035),
                        fl.Triangle("PFar", 0.026, 0.035, 0.044),
                        fl.Triangle("PVeryFar", 0.035, 0.044, math.pi),
                    ],
                ),
            ],
            output_variables=[
                fl.OutputVariable(
                    name="linearVelocity",
                    description="Linear velocity command [m/s]",
                    enabled=True,
                    minimum=self.linear_min,
                    maximum=self.linear_max,
                    lock_range=True,
                    aggregation=fl.AlgebraicSum(),
                    defuzzifier=fl.Centroid(resolution=100),
                    default_value=0.0,
                    lock_previous=False,
                    terms=[
                        fl.Triangle("FastBackwards", -0.80, -0.60, -0.40),
                        fl.Triangle("Backwards", -0.60, -0.40, -0.20),
                        fl.Triangle("MediumBackwards", -0.40, -0.20, 0.000),
                        fl.Triangle("Zero", -0.20, 0.000, 0.20),
                        fl.Triangle("MediumForward", 0.000, 0.20, 0.40),
                        fl.Triangle("Forward", 0.20, 0.40, 0.60),
                        fl.Triangle("FastForward", 0.40, 0.60, 0.80),
                    ],
                ),
                fl.OutputVariable(
                    name="angularVelocity",
                    description="Angular velocity command [rad/s]",
                    enabled=True,
                    minimum=self.angular_min,
                    maximum=self.angular_max,
                    lock_range=True,
                    aggregation=fl.AlgebraicSum(),
                    defuzzifier=fl.Centroid(resolution=100),
                    default_value=0.0,
                    lock_previous=False,
                    terms=[
                        fl.Triangle("FastRight", -2.000, -1.500, -1.000),
                        fl.Triangle("SlowRight", -1.500, -1.000, -0.500),
                        fl.Triangle("VSlowRight", -1.000, -0.500, 0.000),
                        fl.Triangle("Zero", -0.500, 0.000, 0.500),
                        fl.Triangle("VSlowLeft", 0.000, 0.50, 1.000),
                        fl.Triangle("SlowLeft", 0.500, 1.000, 1.500),
                        fl.Triangle("FastLeft", 1.000, 1.500, 2.000),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="mamdani",
                    description="Unified waypoint rule base",
                    enabled=True,
                    conjunction=fl.AlgebraicProduct(),
                    disjunction=fl.AlgebraicSum(),
                    implication=fl.AlgebraicProduct(),
                    activation=fl.General(),
                    rules=[
                        fl.Rule.create("if distanceError is Zero then linearVelocity is Zero and angularVelocity is Zero"),
                        
                        fl.Rule.create("if angleError is NVeryFar then linearVelocity is Zero and angularVelocity is FastRight"),
                        fl.Rule.create("if angleError is NFar then linearVelocity is Zero and angularVelocity is SlowRight"),
                        fl.Rule.create("if angleError is PFar then linearVelocity is Zero and angularVelocity is SlowLeft"),
                        fl.Rule.create("if angleError is PVeryFar then linearVelocity is Zero and angularVelocity is FastLeft"),
                        
                        fl.Rule.create("if angleError is NNear and distanceError is NVeryFar then linearVelocity is FastBackwards and angularVelocity is VSlowRight"),
                        fl.Rule.create("if angleError is NNear and distanceError is NFar then linearVelocity is Backwards and angularVelocity is VSlowRight"),
                        fl.Rule.create("if angleError is NNear and distanceError is NNear then linearVelocity is MediumBackwards and angularVelocity is VSlowRight"),
                        fl.Rule.create("if angleError is NNear and distanceError is PNear then linearVelocity is MediumForward and angularVelocity is VSlowRight"),
                        fl.Rule.create("if angleError is NNear and distanceError is PFar then linearVelocity is Forward and angularVelocity is VSlowRight"),
                        fl.Rule.create("if angleError is NNear and distanceError is PVeryFar then linearVelocity is FastForward and angularVelocity is VSlowRight"),
                        
                        fl.Rule.create("if angleError is PNear and distanceError is NVeryFar then linearVelocity is FastBackwards and angularVelocity is VSlowLeft"),
                        fl.Rule.create("if angleError is PNear and distanceError is NFar then linearVelocity is Backwards and angularVelocity is VSlowLeft"),
                        fl.Rule.create("if angleError is PNear and distanceError is NNear then linearVelocity is MediumBackwards and angularVelocity is VSlowLeft"),
                        fl.Rule.create("if angleError is PNear and distanceError is PNear then linearVelocity is MediumForward and angularVelocity is VSlowLeft"),
                        fl.Rule.create("if angleError is PNear and distanceError is PFar then linearVelocity is Forward and angularVelocity is VSlowLeft"),
                        fl.Rule.create("if angleError is PNear and distanceError is PVeryFar then linearVelocity is FastForward and angularVelocity is VSlowLeft"),
                        
                        fl.Rule.create("if angleError is Zero and distanceError is NVeryFar then linearVelocity is FastBackwards and angularVelocity is Zero"),
                        fl.Rule.create("if angleError is Zero and distanceError is NFar then linearVelocity is Backwards and angularVelocity is Zero"),
                        fl.Rule.create("if angleError is Zero and distanceError is NNear then linearVelocity is MediumBackwards and angularVelocity is Zero"),
                        fl.Rule.create("if angleError is Zero and distanceError is PNear then linearVelocity is MediumForward and angularVelocity is Zero"),
                        fl.Rule.create("if angleError is Zero and distanceError is PFar then linearVelocity is Forward and angularVelocity is Zero"),
                        fl.Rule.create("if angleError is Zero and distanceError is PVeryFar then linearVelocity is FastForward and angularVelocity is Zero"),
],
                )
            ],
        )
        self.engine.rule_block("mamdani").load_rules(self.engine)
        self.last_distance_error = math.inf
        self.last_angle_error = 0.0
        self.last_desired_heading = 0.0

    def compute(self, x, y, theta, target_x, target_y):
        dx = float(target_x) - float(x)
        dy = float(target_y) - float(y)

        distance_error = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        angle_error = wrap_angle(desired_heading - float(theta))

        self.last_distance_error = distance_error
        self.last_angle_error = angle_error
        self.last_desired_heading = desired_heading
        
        print(f"distance_error: {distance_error}, heading_error: {angle_error}")

        # At the waypoint, heading is irrelevant: stop completely.
        if distance_error <= self.goal_tolerance:
            return 0.0, 0.0, True
        distanceError = self.engine.input_variable("distanceError")
        angleError= self.engine.input_variable("angleError")
        
        distanceError.value = distance_error
        angleError.value = angle_error
        
        print("================ DISTANCE MF ================")
        
        for term in distanceError.terms:
            degree = term.membership(distance_error)
            print(" {:12s} -> {:.4f}".format(term.name, float(degree)))
        
        print("================ ANGLE MF ================")
        for term in angleError.terms:
            degree = term.membership(angle_error)
            print(" {:12s} -> {:.4f}".format(term.name, float(degree)))
            
        self.engine.process()
        
        print(f"Fuzzy distance error: {distanceError.value}")
        print(f"Fuzzy angle error: {angleError.value}")
        
        print("================ RULE BLOCK ================")
        rule_block = self.engine.rule_block("mamdani").rules
        for i, rule in enumerate(rule_block):
            print(" R{:02d}: activation={} triggered={}".format(i, rule.activation_degree, rule.triggered))
        
        linVel = self.engine.output_variable("linearVelocity")
        angVel = self.engine.output_variable("angularVelocity")
        
        print(f"Fuzzy Output Value Linear: {linVel.fuzzy_value()}")
        print(f"Fuzzy Output Value Angular: {angVel.fuzzy_value()}")
        linear = float(linVel.value)
        angular = float(angVel.value)

        if not math.isfinite(linear):
            linear = 0.0
        if not math.isfinite(angular):
            angular = 0.0

        return linear, angular, False

    def reset(self):
        self.engine.restart()
        self.last_distance_error = math.inf
        self.last_angle_error = 0.0
        self.last_desired_heading = 0.0
