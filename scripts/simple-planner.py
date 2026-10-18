#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple waypoint planner for the SARA controlled localisation experiments.

The controller intentionally remains simple and outside the scientific scope of
localisation. Two independent PD controllers are used:

  * FORWARD:    distance PD -> linear velocity, angular velocity = 0
  * ORIENTING:  heading PD  -> angular velocity, linear velocity = 0

A hysteresis band avoids rapid switching between the two control states:
ORIENTING is entered when |heading_error| > theta_1 and FORWARD is resumed when
|heading_error| <= theta_tolerance, with theta_tolerance < theta_1.

Both velocity commands are rate limited to provide smooth acceleration and
braking. Waypoints marked as checkpoints stop the wheelchair and require SPACE
or ENTER in the terminal before continuing. At each checkpoint the ROS
timestamp and arrival pose are logged, and a ground-truth template row is
written automatically for later manual completion of gt_dx and gt_dy.
"""

import csv
import math
import os
import select
import sys
import termios
import threading
import tty

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
    FORWARD = "FORWARD"
    ORIENTING = "ORIENTING"
    CHECKPOINT = "CHECKPOINT"
    DONE = "DONE"
    STOPPED = "STOPPED"

    def __init__(self):
        # ------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------
        # A fixed localisation estimator is intentionally used as feedback so
        # the experimental trajectory itself is repeatable.
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/pose/filtered/ukf/incremental"
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.checkpoint_topic = rospy.get_param(
            "~checkpoint_topic", "/simple_planner/checkpoint_reached"
        )
        self.target_topic = rospy.get_param(
            "~target_topic", "/simple_planner/target_pose"
        )

        # ------------------------------------------------------------
        # Two PD controllers
        # ------------------------------------------------------------
        # u = Kp*e + Kd*de/dt
        self.kp_linear = float(rospy.get_param("~kp_linear", 0.34))
        self.kd_linear = float(rospy.get_param("~kd_linear", 0.0))

        self.kp_angular = float(rospy.get_param("~kp_angular", 0.45))
        self.kd_angular = float(rospy.get_param("~kd_angular", 0.0))

        # State-switching thresholds.
        self.d_tolerance = float(rospy.get_param("~d_tolerance", 0.05))
        self.theta_tolerance = float(
            rospy.get_param("~theta_tolerance", 0.08)
        )
        self.theta_1 = float(rospy.get_param("~theta_1", 0.20))

        if self.d_tolerance <= 0.0:
            raise rospy.ROSException("d_tolerance must be > 0")
        if self.theta_tolerance <= 0.0:
            raise rospy.ROSException("theta_tolerance must be > 0")
        if self.theta_1 <= self.theta_tolerance:
            raise rospy.ROSException(
                "theta_1 must be greater than theta_tolerance"
            )

        # ------------------------------------------------------------
        # Velocity limits and command ramps
        # ------------------------------------------------------------
        self.min_linear_velocity = float(
            rospy.get_param("~min_linear_velocity", 0.08)
        )
        self.max_linear_velocity = float(
            rospy.get_param("~max_linear_velocity", 0.25)
        )
        self.min_angular_velocity = float(
            rospy.get_param("~min_angular_velocity", 0.10)
        )
        self.max_angular_velocity = float(
            rospy.get_param("~max_angular_velocity", 0.45)
        )

        # Maximum command variation per second.
        self.linear_acceleration = float(
            rospy.get_param("~linear_acceleration", 0.10)
        )
        self.angular_acceleration = float(
            rospy.get_param("~angular_acceleration", 0.50)
        )

        if self.min_linear_velocity < 0.0:
            raise rospy.ROSException("min_linear_velocity must be >= 0")
        if self.min_angular_velocity < 0.0:
            raise rospy.ROSException("min_angular_velocity must be >= 0")
        if self.min_linear_velocity > self.max_linear_velocity:
            raise rospy.ROSException(
                "min_linear_velocity cannot exceed max_linear_velocity"
            )
        if self.min_angular_velocity > self.max_angular_velocity:
            raise rospy.ROSException(
                "min_angular_velocity cannot exceed max_angular_velocity"
            )
        if self.linear_acceleration <= 0.0:
            raise rospy.ROSException("linear_acceleration must be > 0")
        if self.angular_acceleration <= 0.0:
            raise rospy.ROSException("angular_acceleration must be > 0")

        # ------------------------------------------------------------
        # Runtime / experiment parameters
        # ------------------------------------------------------------
        self.control_hz = float(rospy.get_param("~control_hz", 20.0))
        self.pose_timeout = float(rospy.get_param("~pose_timeout", 0.50))
        self.run_id = int(rospy.get_param("~run_id", 1))

        # Default trajectory only acts as a safe fallback. Normal operation
        # loads the selected YAML from simple_planner.launch.
        default_waypoints = [
            {"name": "P1", "x": 3.0, "y": 0.0, "checkpoint": True},
            {"name": "P2", "x": 3.0, "y": 3.5, "checkpoint": True},
            {"name": "P3", "x": 0.0, "y": 3.5, "checkpoint": True},
            {"name": "P4", "x": 0.0, "y": 0.0, "checkpoint": True},
        ]

        raw_waypoints = rospy.get_param("~waypoints", default_waypoints)
        if len(raw_waypoints) < 1:
            raise rospy.ROSException("At least one waypoint is required")

        self.waypoints = []
        for i, wp in enumerate(raw_waypoints):
            self.waypoints.append(
                {
                    "name": str(wp.get("name", "P{}".format(i + 1))),
                    "x": float(wp["x"]),
                    "y": float(wp["y"]),
                    "checkpoint": bool(wp.get("checkpoint", False)),
                }
            )

        # ------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------
        self.log_path = rospy.get_param(
            "~log_path", "/tmp/simple_planner_events.csv"
        )
        self.ground_truth_path = rospy.get_param(
            "~ground_truth_path", "/tmp/ground_truth.csv"
        )

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------
        self.pose = None
        self.last_pose_rx_time = None

        self.state = self.WAITING
        self.running = False
        self.target_index = 0

        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_control_time = None
        self.prev_distance_error = None
        self.prev_heading_error = None

        self.checkpoint_continue = threading.Event()
        self.keyboard_thread = None

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

        # Start/stop are retained for convenience and safety. Checkpoint
        # continuation is deliberately keyboard-only.
        self.start_srv = rospy.Service("~start", Trigger, self.start_callback)
        self.stop_srv = rospy.Service("~stop", Trigger, self.stop_callback)

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_hz), self.control_callback
        )

        self.prepare_log()
        self.prepare_ground_truth_log()
        self.start_keyboard_thread()
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("sara_simple_planner ready")
        rospy.loginfo("Pose feedback: %s", self.pose_topic)
        rospy.loginfo("Velocity output: %s", self.cmd_vel_topic)
        rospy.loginfo(
            "Route: %s",
            " -> ".join(
                "{}({:.3f},{:.3f}){}".format(
                    w["name"],
                    w["x"],
                    w["y"],
                    "*" if w["checkpoint"] else "",
                )
                for w in self.waypoints
            ),
        )
        rospy.loginfo(
            "PD: linear Kp=%.3f Kd=%.3f | angular Kp=%.3f Kd=%.3f",
            self.kp_linear,
            self.kd_linear,
            self.kp_angular,
            self.kd_angular,
        )
        rospy.loginfo(
            "Thresholds: d_tol=%.3f m | theta_tol=%.3f rad | theta_1=%.3f rad",
            self.d_tolerance,
            self.theta_tolerance,
            self.theta_1,
        )
        rospy.loginfo(
            "Limits: v=[%.3f, %.3f] m/s | w=[%.3f, %.3f] rad/s",
            self.min_linear_velocity,
            self.max_linear_velocity,
            self.min_angular_velocity,
            self.max_angular_velocity,
        )
        rospy.loginfo(
            "Ramps: linear=%.3f m/s^2 | angular=%.3f rad/s^2",
            self.linear_acceleration,
            self.angular_acceleration,
        )

    # ------------------------------------------------------------
    # ROS callbacks / services
    # ------------------------------------------------------------

    def pose_callback(self, msg):
        self.pose = msg
        self.last_pose_rx_time = rospy.Time.now()

        # Preserve the original behaviour: start automatically once the first
        # valid feedback pose is available.
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
        return TriggerResponse(success=True, message="Planner stopped")

    # ------------------------------------------------------------
    # Experiment state machine
    # ------------------------------------------------------------

    def reset_experiment(self):
        self.target_index = 0
        self.state = self.WAITING
        self.checkpoint_continue.clear()
        self.reset_controller()

    def start_motion(self):
        self.running = True
        self.state = self.FORWARD
        self.target_index = 0
        self.checkpoint_continue.clear()
        self.reset_controller()
        self.publish_target()
        self.log_event("START")

        rospy.loginfo(
            "Starting run %d from current pose. First target: %s",
            self.run_id,
            self.waypoints[self.target_index]["name"],
        )

    def control_callback(self, _event):
        if not self.running:
            self.publish_zero()
            return

        if self.pose is None or self.last_pose_rx_time is None:
            self.publish_zero()
            return

        now = rospy.Time.now()
        pose_age = (now - self.last_pose_rx_time).to_sec()
        if pose_age > self.pose_timeout:
            rospy.logwarn_throttle(
                1.0,
                "Pose feedback is stale (%.3f s). Stopping command.",
                pose_age,
            )
            # Safety stop is immediate, not ramped.
            self.publish_zero(reset_history=True)
            return

        if self.state == self.FORWARD:
            self.control_forward(now)
        elif self.state == self.ORIENTING:
            self.control_orienting(now)
        elif self.state == self.CHECKPOINT:
            self.control_checkpoint()
        elif self.state in (self.DONE, self.STOPPED, self.WAITING):
            self.publish_zero()

    def target_errors(self):
        target = self.waypoints[self.target_index]
        dx = target["x"] - self.pose.x
        dy = target["y"] - self.pose.y
        distance_error = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - self.pose.theta)
        return distance_error, heading_error

    def control_forward(self, now):
        """Distance PD. Angular command is always zero in this state."""
        distance_error, heading_error = self.target_errors()
        dt = self.controller_dt(now)

        # Target reached: brake smoothly before declaring the waypoint reached.
        if distance_error <= self.d_tolerance:
            linear = move_towards(
                self.last_linear_cmd,
                0.0,
                self.linear_acceleration * dt,
            )
            self.publish_cmd(linear, 0.0)

            if abs(linear) <= 1e-6:
                self.waypoint_reached()
            return

        # Heading has drifted outside the outer threshold. Brake linear motion
        # first, then enter ORIENTING from zero linear velocity.
        if abs(heading_error) > self.theta_1:
            linear = move_towards(
                self.last_linear_cmd,
                0.0,
                self.linear_acceleration * dt,
            )
            self.publish_cmd(linear, 0.0)

            if abs(linear) <= 1e-6:
                self.state = self.ORIENTING
                self.prev_heading_error = None
                self.last_angular_cmd = 0.0
                rospy.loginfo(
                    "State -> ORIENTING | heading error=%.3f rad",
                    heading_error,
                )
            return

        # Normal forward motion.
        requested = self.pd_command(
            distance_error,
            self.prev_distance_error,
            self.kp_linear,
            self.kd_linear,
            dt,
        )
        requested = self.limit_positive_command(
            requested,
            self.min_linear_velocity,
            self.max_linear_velocity,
        )

        linear = move_towards(
            self.last_linear_cmd,
            requested,
            self.linear_acceleration * dt,
        )

        self.prev_distance_error = distance_error
        self.prev_heading_error = None
        self.publish_cmd(linear, 0.0)

    def control_orienting(self, now):
        """Heading PD. Linear command is always zero in this state."""
        _distance_error, heading_error = self.target_errors()
        dt = self.controller_dt(now)

        # Once inside the inner angular tolerance, brake angular motion to zero
        # before returning to FORWARD.
        if abs(heading_error) <= self.theta_tolerance:
            angular = move_towards(
                self.last_angular_cmd,
                0.0,
                self.angular_acceleration * dt,
            )
            self.publish_cmd(0.0, angular)

            if abs(angular) <= 1e-6:
                self.state = self.FORWARD
                self.prev_distance_error = None
                self.prev_heading_error = None
                self.last_linear_cmd = 0.0
                rospy.loginfo(
                    "State -> FORWARD | heading error=%.3f rad",
                    heading_error,
                )
            return

        requested = self.pd_command(
            heading_error,
            self.prev_heading_error,
            self.kp_angular,
            self.kd_angular,
            dt,
        )
        requested = self.limit_signed_command(
            requested,
            heading_error,
            self.min_angular_velocity,
            self.max_angular_velocity,
        )

        angular = move_towards(
            self.last_angular_cmd,
            requested,
            self.angular_acceleration * dt,
        )

        self.prev_heading_error = heading_error
        self.prev_distance_error = None
        self.publish_cmd(0.0, angular)

    def waypoint_reached(self):
        target = self.waypoints[self.target_index]
        self.publish_zero(reset_history=True)

        if target["checkpoint"]:
            self.enter_checkpoint()
            return

        self.log_event("WAYPOINT_REACHED")
        rospy.loginfo(
            "Waypoint %s reached (no checkpoint).",
            target["name"],
        )
        self.advance_to_next_waypoint()

    def enter_checkpoint(self):
        target = self.waypoints[self.target_index]
        timestamp = rospy.Time.now().to_sec()

        self.state = self.CHECKPOINT
        self.checkpoint_continue.clear()
        self.publish_zero(reset_history=True)

        event = (
            "run={run},timestamp={ts:.9f},waypoint={wp},"
            "ref_x={x:.8f},ref_y={y:.8f},"
            "pose_x={px:.8f},pose_y={py:.8f},pose_theta={pth:.8f}"
        ).format(
            run=self.run_id,
            ts=timestamp,
            wp=target["name"],
            x=target["x"],
            y=target["y"],
            px=self.pose.x,
            py=self.pose.y,
            pth=self.pose.theta,
        )

        self.checkpoint_pub.publish(String(data=event))
        self.log_event("CHECKPOINT", timestamp=timestamp)
        self.log_ground_truth_checkpoint(timestamp, target)

        rospy.loginfo("====================================================")
        rospy.loginfo("CHECKPOINT %s", target["name"])
        rospy.loginfo("timestamp = %.9f", timestamp)
        rospy.loginfo(
            "pose      = (%.6f, %.6f, %.6f)",
            self.pose.x,
            self.pose.y,
            self.pose.theta,
        )
        rospy.loginfo(
            "reference = (%.6f, %.6f)",
            target["x"],
            target["y"],
        )
        rospy.loginfo("Press SPACE or ENTER to continue...")
        rospy.loginfo("====================================================")

    def control_checkpoint(self):
        self.publish_zero()

        if not self.checkpoint_continue.is_set():
            return

        self.checkpoint_continue.clear()
        self.advance_to_next_waypoint()

    def advance_to_next_waypoint(self):
        if self.target_index >= len(self.waypoints) - 1:
            self.finish_experiment()
            return

        self.target_index += 1
        self.state = self.FORWARD
        self.reset_controller()
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

        target = self.waypoints[-1]
        rospy.loginfo(
            "Run %d completed at final waypoint %s (%.3f, %.3f).",
            self.run_id,
            target["name"],
            target["x"],
            target["y"],
        )

    # ------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------

    def start_keyboard_thread(self):
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            name="simple_planner_keyboard",
            daemon=True,
        )
        self.keyboard_thread.start()

    def keyboard_loop(self):
        """Read SPACE/ENTER directly from the controlling terminal."""
        stream = None
        old_settings = None

        try:
            try:
                stream = open("/dev/tty", "r")
            except OSError:
                stream = sys.stdin

            if not stream.isatty():
                rospy.logerr(
                    "No interactive terminal is available. "
                    "SPACE/ENTER checkpoint continuation cannot be read."
                )
                return

            fd = stream.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)

            while not rospy.is_shutdown():
                readable, _, _ = select.select([stream], [], [], 0.20)
                if not readable:
                    continue

                char = stream.read(1)
                if char not in (" ", "\n", "\r"):
                    continue

                if self.state == self.CHECKPOINT:
                    self.checkpoint_continue.set()
                    rospy.loginfo("Continue key received.")

        except Exception as exc:
            rospy.logerr("Keyboard input failed: %s", exc)
        finally:
            if stream is not None and old_settings is not None:
                try:
                    termios.tcsetattr(
                        stream.fileno(), termios.TCSADRAIN, old_settings
                    )
                except Exception:
                    pass
            if stream is not None and stream is not sys.stdin:
                try:
                    stream.close()
                except Exception:
                    pass

    # ------------------------------------------------------------
    # Controller helpers
    # ------------------------------------------------------------

    def pd_command(self, error, previous_error, kp, kd, dt):
        derivative = 0.0
        if previous_error is not None and dt > 1e-9:
            derivative = (error - previous_error) / dt
        return kp * error + kd * derivative

    def limit_positive_command(self, command, minimum, maximum):
        if command <= 0.0:
            return 0.0
        if command < minimum:
            return minimum
        return min(command, maximum)

    def limit_signed_command(self, command, error, minimum, maximum):
        if abs(command) <= 1e-12:
            return 0.0
        magnitude = clamp(abs(command), minimum, maximum)
        return math.copysign(magnitude, error)

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

    # ------------------------------------------------------------
    # ROS command / target output
    # ------------------------------------------------------------

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
            self.prev_distance_error = None
            self.prev_heading_error = None

    def publish_target(self):
        target = self.waypoints[self.target_index]

        msg = Pose2D()
        msg.x = target["x"]
        msg.y = target["y"]
        msg.theta = 0.0
        self.target_pub.publish(msg)

        rospy.loginfo(
            "New target: %s | x=%.3f y=%.3f | checkpoint=%s",
            target["name"],
            target["x"],
            target["y"],
            target["checkpoint"],
        )

    # ------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------

    @staticmethod
    def ensure_parent(path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def prepare_log(self):
        self.ensure_parent(self.log_path)
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
                    "checkpoint",
                    "ref_x",
                    "ref_y",
                    "pose_x",
                    "pose_y",
                    "pose_theta",
                ]
            )
            self.log_file.flush()

    def prepare_ground_truth_log(self):
        self.ensure_parent(self.ground_truth_path)
        new_file = not os.path.exists(self.ground_truth_path)

        self.gt_file = open(self.ground_truth_path, "a", newline="")
        self.gt_writer = csv.writer(self.gt_file)

        if new_file:
            self.gt_writer.writerow(
                [
                    "run_id",
                    "timestamp",
                    "checkpoint_id",
                    "ref_x",
                    "ref_y",
                    "gt_dx",
                    "gt_dy",
                    "gt_x",
                    "gt_y",
                ]
            )
            self.gt_file.flush()

    def log_ground_truth_checkpoint(self, timestamp, target):
        self.gt_writer.writerow(
            [
                self.run_id,
                "{:.9f}".format(timestamp),
                target["name"],
                "{:.8f}".format(target["x"]),
                "{:.8f}".format(target["y"]),
                "",
                "",
                "",
                "",
            ]
        )
        self.gt_file.flush()

    def log_event(self, event, timestamp=None):
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

        if timestamp is None:
            timestamp = rospy.Time.now().to_sec()

        self.log_writer.writerow(
            [
                "{:.9f}".format(timestamp),
                event,
                self.state,
                self.run_id,
                target["name"],
                int(target["checkpoint"]),
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

        try:
            self.gt_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("simple_planner")

    try:
        planner = SimplePlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
