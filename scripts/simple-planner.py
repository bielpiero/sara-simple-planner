#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os

import rospy
from geometry_msgs.msg import Pose2D, Twist
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, low, high):
    return max(low, min(high, value))


def move_towards(current, target, max_delta):
    """Rate-limit a command without overshooting the requested target."""
    if target > current:
        return min(current + max_delta, target)
    if target < current:
        return max(current - max_delta, target)
    return target


class SimplePlanner:
    WAITING = "WAITING"
    GO_TO_POSITION = "GO_TO_POSITION"
    CHECKPOINT = "CHECKPOINT"
    DONE = "DONE"
    STOPPED = "STOPPED"

    def __init__(self):
        # ------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------
        # Raw wheel odometry is intentionally used as planner feedback so
        # trajectory generation does not depend on any estimator under test.
        self.pose_topic = rospy.get_param("~pose_topic", "/pose/raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.checkpoint_topic = rospy.get_param(
            "~checkpoint_topic", "/simple_planner/checkpoint_reached"
        )
        self.target_topic = rospy.get_param(
            "~target_topic", "/simple_planner/target_pose"
        )

        # ------------------------------------------------------------
        # Controller
        # ------------------------------------------------------------
        # Discrete PD:
        #   u(k) = Kp*e(k) + Kd*(e(k)-e(k-1))
        #   Kd   = Kp*Td/dt
        self.kp_linear = rospy.get_param("~kp_linear", 0.45)
        self.td_linear = rospy.get_param("~td_linear", 0.08)

        self.kp_angular = rospy.get_param("~kp_angular", 1.00)
        self.td_angular = rospy.get_param("~td_angular", 0.06)

        self.derivative_limit_ratio = rospy.get_param(
            "~derivative_limit_ratio", 0.80
        )

        self.position_tolerance = rospy.get_param(
            "~position_tolerance", 0.03
        )

        self.max_linear_velocity = rospy.get_param(
            "~max_linear_velocity", 0.20
        )
        self.max_angular_velocity = rospy.get_param(
            "~max_angular_velocity", 1.00
        )

        self.min_effective_linear_velocity = rospy.get_param(
            "~min_effective_linear_velocity", 0.3
        )
        
        self.min_effective_angular_velocity = rospy.get_param(
            "~min_effective_angular_velocity", 0.3
        )

        self.acceleration_distance = rospy.get_param(
            "~acceleration_distance", 0.50
        )
        self.deceleration_distance = rospy.get_param(
            "~deceleration_distance", 0.50
        )

        if self.acceleration_distance <= 0.0:
            raise rospy.ROSException("acceleration_distance must be > 0")
        if self.deceleration_distance <= 0.0:
            raise rospy.ROSException("deceleration_distance must be > 0")

        self.linear_acceleration = (
            self.max_linear_velocity * self.max_linear_velocity
            / (2.0 * self.acceleration_distance)
        )
        self.linear_deceleration = (
            self.max_linear_velocity * self.max_linear_velocity
            / (2.0 * self.deceleration_distance)
        )

        self.angular_acceleration = rospy.get_param(
            "~angular_acceleration", 0.40
        )
        self.angular_deceleration = rospy.get_param(
            "~angular_deceleration", 0.80
        )

        self.rotate_in_place_threshold = rospy.get_param(
            "~rotate_in_place_threshold", 0.30
        )

        self.control_hz = rospy.get_param("~control_hz", 20.0)
        self.pose_timeout = rospy.get_param("~pose_timeout", 0.50)

        # ------------------------------------------------------------
        # Experiment
        # ------------------------------------------------------------
        # Five experimental repetitions should be run independently. Set a
        # different run_id for each execution if desired.
        self.run_id = int(rospy.get_param("~run_id", 1))

        self.dwell_time = rospy.get_param("~dwell_time", 2.0)
        self.manual_checkpoint = rospy.get_param("~manual_checkpoint", False)

        # Default square trajectory:
        # START=(0,0) -> P1=(3,0) -> P2=(3,3.5)
        # -> P3=(0,3.5) -> HOME=(0,0)
        default_waypoints = [
            {"name": "P1", "marker_id": -1, "x": 3.0, "y": 0.0},
            {"name": "P2", "marker_id": -1, "x": 3.0, "y": 3.5},
            {"name": "P3", "marker_id": -1, "x": 0.0, "y": 3.5},
            {"name": "HOME", "marker_id": -1, "x": 0.0, "y": 0.0},
        ]

        raw_waypoints = rospy.get_param("~waypoints", default_waypoints)
        if len(raw_waypoints) < 1:
            raise rospy.ROSException("At least one waypoint is required")

        self.waypoints = []
        for i, wp in enumerate(raw_waypoints):
            self.waypoints.append(
                {
                    "name": str(wp.get("name", "P{}".format(i + 1))),
                    "marker_id": int(wp.get("marker_id", -1)),
                    "x": float(wp["x"]),
                    "y": float(wp["y"]),
                }
            )

        # CSV event log
        self.log_path = rospy.get_param(
            "~log_path", "/tmp/simple_planner_events.csv"
        )

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.pose = None
        self.last_pose_rx_time = None

        self.state = self.WAITING
        self.running = False
        self.target_index = 0

        self.checkpoint_start_time = None
        self.waiting_manual_continue = False

        # Last published command, used by acceleration/deceleration ramps.
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_control_time = None

        # Previous controller errors for the discrete derivative term.
        self.prev_distance_error = None
        self.prev_heading_error = None

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------
        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1
        )
        self.checkpoint_pub = rospy.Publisher(
            self.checkpoint_topic, String, queue_size=10
        )
        self.target_pub = rospy.Publisher(
            self.target_topic, Pose2D, queue_size=1, latch=True
        )

        self.pose_sub = rospy.Subscriber(
            self.pose_topic, Pose2D, self.pose_callback, queue_size=1
        )

        self.start_srv = rospy.Service(
            "~start", Trigger, self.start_callback
        )
        self.stop_srv = rospy.Service(
            "~stop", Trigger, self.stop_callback
        )
        self.continue_srv = rospy.Service(
            "~continue", Trigger, self.continue_callback
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_hz),
            self.control_callback
        )

        self.prepare_log()
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("sara_simple_planner ready")
        rospy.loginfo("Pose feedback: %s", self.pose_topic)
        rospy.loginfo("Velocity output: %s", self.cmd_vel_topic)
        rospy.loginfo(
            "Route: %s",
            " -> ".join(
                "{}({:.2f},{:.2f})".format(
                    w["name"], w["x"], w["y"]
                )
                for w in self.waypoints
            ),
        )
        rospy.loginfo(
            "Position tolerance: %.3f m | rotate threshold: %.3f rad",
            self.position_tolerance,
            self.rotate_in_place_threshold,
        )
        rospy.loginfo(
            "PD controller: Kp_lin=%.3f Td_lin=%.3f s | "
            "Kp_ang=%.3f Td_ang=%.3f s",
            self.kp_linear,
            self.td_linear,
            self.kp_angular,
            self.td_angular,
        )
        rospy.loginfo(
            "Linear profile: v_max=%.3f m/s | accel_dist=%.3f m "
            "(a=%.4f m/s^2) | decel_dist=%.3f m (a=%.4f m/s^2)",
            self.max_linear_velocity,
            self.acceleration_distance,
            self.linear_acceleration,
            self.deceleration_distance,
            self.linear_deceleration,
        )

    # ------------------------------------------------------------
    # ROS callbacks / services
    # ------------------------------------------------------------

    def pose_callback(self, msg):
        self.pose = msg
        self.last_pose_rx_time = rospy.Time.now()

        # Start once the first valid pose has been received.
        if not self.running and self.state == self.WAITING:
            self.start_motion()

    def start_callback(self, _req):
        if self.pose is None:
            return TriggerResponse(
                success=False,
                message="No pose received yet on {}".format(self.pose_topic),
            )

        if self.running:
            return TriggerResponse(
                success=False,
                message="Planner is already running",
            )

        if self.state in (self.DONE, self.STOPPED):
            self.reset_experiment()

        self.start_motion()

        return TriggerResponse(
            success=True,
            message="Planner started. Target: {}".format(
                self.waypoints[self.target_index]["name"]
            ),
        )

    def stop_callback(self, _req):
        self.running = False
        self.state = self.STOPPED
        self.publish_zero(reset_history=True)
        self.log_event("STOPPED")

        return TriggerResponse(
            success=True,
            message="Planner stopped",
        )

    def continue_callback(self, _req):
        if not self.waiting_manual_continue:
            return TriggerResponse(
                success=False,
                message="Planner is not waiting at a checkpoint",
            )

        self.waiting_manual_continue = False
        self.advance_after_checkpoint()

        return TriggerResponse(
            success=True,
            message="Continuing experiment",
        )

    # ------------------------------------------------------------
    # Experiment state machine
    # ------------------------------------------------------------

    def reset_experiment(self):
        self.target_index = 0
        self.waiting_manual_continue = False
        self.checkpoint_start_time = None
        self.state = self.WAITING
        self.reset_controller()

    def start_motion(self):
        self.running = True
        self.state = self.GO_TO_POSITION
        self.reset_controller()
        self.publish_target()
        self.log_event("START")

        rospy.loginfo(
            "Starting run %d. First target: %s",
            self.run_id,
            self.waypoints[self.target_index]["name"],
        )

    def control_callback(self, _event):
        if not self.running:
            self.publish_zero()
            return

        if self.pose is None:
            self.publish_zero()
            return

        now = rospy.Time.now()

        if self.last_pose_rx_time is None:
            self.publish_zero()
            return

        pose_age = (now - self.last_pose_rx_time).to_sec()
        if pose_age > self.pose_timeout:
            rospy.logwarn_throttle(
                1.0,
                "Pose feedback is stale (%.3f s). Stopping command.",
                pose_age,
            )
            self.publish_zero(reset_history=True)
            return

        if self.state == self.GO_TO_POSITION:
            self.control_position(now)
        elif self.state == self.CHECKPOINT:
            self.control_checkpoint(now)
        elif self.state in (self.DONE, self.STOPPED, self.WAITING):
            self.publish_zero()

    def control_position(self, now):
        target = self.waypoints[self.target_index]

        dx = target["x"] - self.pose.x
        dy = target["y"] - self.pose.y
        distance_error = math.hypot(dx, dy)

        # The only normal full-stop condition during navigation is reaching
        # the positional tolerance.
        if distance_error <= self.position_tolerance:
            self.publish_zero(reset_history=True)

            if self.target_index == len(self.waypoints) - 1:
                self.log_event("HOME_REACHED")
                self.finish_experiment()
            else:
                self.enter_checkpoint()
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - self.pose.theta)

        dt = self.controller_dt(now)

        linear_pd = self.pd_command(
            distance_error,
            self.prev_distance_error,
            self.kp_linear,
            self.td_linear,
            dt,
        )

        angular_pd = self.pd_command(
            heading_error,
            self.prev_heading_error,
            self.kp_angular,
            self.td_angular,
            dt,
        )

        self.prev_distance_error = distance_error
        self.prev_heading_error = heading_error

        angular_target = clamp(
            angular_pd,
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )
        
        if abs(angular_target) < self.min_effective_angular_velocity:
            angular_target = self.min_effective_angular_velocity if angular_target >= 0 else -self.min_effective_angular_velocity

        angular_rate = (
            self.angular_acceleration
            if abs(angular_target) >= abs(self.last_angular_cmd)
            else self.angular_deceleration
        )

        angular = move_towards(
            self.last_angular_cmd,
            angular_target,
            angular_rate * dt,
        )

        # Braking envelope:
        # v <= sqrt(2*a_dec*(distance - tolerance))
        # This reaches zero exactly at the waypoint acceptance boundary.
        remaining_to_stop = max(
            distance_error - self.position_tolerance,
            0.0,
        )

        braking_limit = math.sqrt(
            2.0 * self.linear_deceleration * remaining_to_stop
        )
        braking_limit = min(
            braking_limit,
            self.max_linear_velocity,
        )

        if abs(heading_error) >= self.rotate_in_place_threshold:
            linear_target = 0.0
        else:
            linear_target = clamp(
                linear_pd,
                0.0,
                self.max_linear_velocity,
            )
            linear_target = min(
                linear_target,
                braking_limit,
            )

            if distance_error > self.position_tolerance and linear_target < self.min_effective_linear_velocity:
                linear_target = self.min_effective_linear_velocity

        linear_rate = (
            self.linear_acceleration
            if linear_target >= self.last_linear_cmd
            else self.linear_deceleration
        )

        linear = move_towards(
            self.last_linear_cmd,
            linear_target,
            linear_rate * dt,
        )

        self.publish_cmd(linear, angular)

    def enter_checkpoint(self):
        self.state = self.CHECKPOINT
        self.checkpoint_start_time = rospy.Time.now()
        self.waiting_manual_continue = self.manual_checkpoint
        self.publish_zero(reset_history=True)

        target = self.waypoints[self.target_index]

        event = (
            "run={run},waypoint={wp},marker_id={marker},"
            "ref_x={x:.8f},ref_y={y:.8f},"
            "pose_x={px:.8f},pose_y={py:.8f},pose_theta={pth:.8f}"
        ).format(
            run=self.run_id,
            wp=target["name"],
            marker=target["marker_id"],
            x=target["x"],
            y=target["y"],
            px=self.pose.x,
            py=self.pose.y,
            pth=self.pose.theta,
        )

        self.checkpoint_pub.publish(String(data=event))
        self.log_event("CHECKPOINT")

        rospy.loginfo(
            "Checkpoint %s reached | run=%d | "
            "position=(%.3f, %.3f) | arrival theta=%.3f rad",
            target["name"],
            self.run_id,
            self.pose.x,
            self.pose.y,
            self.pose.theta,
        )

        if self.manual_checkpoint:
            rospy.loginfo(
                "Waiting for manual measurement. Continue with: "
                "rosservice call /simple_planner/continue"
            )

    def control_checkpoint(self, now):
        self.publish_zero()

        if self.waiting_manual_continue:
            return

        if self.checkpoint_start_time is None:
            self.checkpoint_start_time = now

        elapsed = (now - self.checkpoint_start_time).to_sec()

        if elapsed >= self.dwell_time:
            self.advance_after_checkpoint()

    def advance_after_checkpoint(self):
        self.checkpoint_start_time = None
        self.reset_controller()

        if self.target_index >= len(self.waypoints) - 1:
            self.finish_experiment()
            return

        self.target_index += 1
        self.state = self.GO_TO_POSITION
        self.publish_target()

        rospy.loginfo(
            "Next target: %s",
            self.waypoints[self.target_index]["name"],
        )

    def finish_experiment(self):
        self.running = False
        self.state = self.DONE
        self.publish_zero(reset_history=True)
        self.log_event("DONE")

        rospy.loginfo(
            "Run %d completed. HOME reached.",
            self.run_id,
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def pd_command(self, error, previous_error, kp, td, dt):
        """
        Discrete PD:
            u(k) = Kp*e(k) + Kd*(e(k)-e(k-1))
            Kd   = Kp*Td/dt

        The D term is limited so it damps the response without cancelling
        the P term while a non-zero error remains.
        """
        p_term = kp * error

        if previous_error is None or dt <= 1e-9 or td <= 0.0:
            return p_term

        kd = kp * td / dt
        d_term = kd * (error - previous_error)

        d_limit = self.derivative_limit_ratio * abs(p_term)
        d_term = clamp(d_term, -d_limit, d_limit)

        return p_term + d_term

    def controller_dt(self, now):
        if self.last_control_time is None:
            dt = 1.0 / self.control_hz
        else:
            dt = (now - self.last_control_time).to_sec()
            if dt <= 0.0:
                dt = 1.0 / self.control_hz

        self.last_control_time = now
        return dt

    def reset_controller(self):
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_control_time = None
        self.prev_distance_error = None
        self.prev_heading_error = None

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)

        self.last_linear_cmd = linear
        self.last_angular_cmd = angular

    def publish_zero(self, reset_history=False):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

        if reset_history:
            self.last_linear_cmd = 0.0
            self.last_angular_cmd = 0.0
            self.last_control_time = None

    def publish_target(self):
        target = self.waypoints[self.target_index]

        msg = Pose2D()
        msg.x = target["x"]
        msg.y = target["y"]

        # Theta has no waypoint-control meaning in this planner. Pose2D is
        # retained only for compatibility with the existing target topic.
        msg.theta = 0.0

        self.target_pub.publish(msg)

        rospy.loginfo(
            "New target: %s | ArUco %d | x=%.3f y=%.3f",
            target["name"],
            target["marker_id"],
            target["x"],
            target["y"],
        )

    def prepare_log(self):
        directory = os.path.dirname(self.log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        new_file = not os.path.exists(self.log_path)

        self.log_file = open(self.log_path, "a", newline="")
        self.log_writer = csv.writer(self.log_file)

        if new_file:
            self.log_writer.writerow(
                [
                    "timestamp",
                    "event",
                    "state",
                    "run_id",
                    "waypoint",
                    "marker_id",
                    "ref_x",
                    "ref_y",
                    "pose_x",
                    "pose_y",
                    "pose_theta",
                ]
            )
            self.log_file.flush()

    def log_event(self, event):
        if not self.waypoints:
            return

        target = self.waypoints[
            min(self.target_index, len(self.waypoints) - 1)
        ]

        px = py = ptheta = float("nan")
        if self.pose is not None:
            px = self.pose.x
            py = self.pose.y
            ptheta = self.pose.theta

        self.log_writer.writerow(
            [
                rospy.Time.now().to_sec(),
                event,
                self.state,
                self.run_id,
                target["name"],
                target["marker_id"],
                target["x"],
                target["y"],
                px,
                py,
                ptheta,
            ]
        )
        self.log_file.flush()

    def shutdown(self):
        self.publish_zero(reset_history=True)

        try:
            self.log_event("SHUTDOWN")
        except Exception:
            pass

        try:
            self.log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("simple_planner")

    try:
        planner = SimplePlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
