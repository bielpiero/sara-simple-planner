#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Waypoint planner for the SARA controlled localisation experiments.

The planner deliberately separates trajectory supervision from control:
  * ORIENTING: a fuzzy heading controller commands angular velocity, v = 0.
  * FORWARD:   a fuzzy distance controller commands linear velocity, w = 0.

A 3 deg / 7 deg hysteresis band prevents state chattering.  Final velocity
commands are rate-limited only as an actuator/comfort constraint.  Checkpoints
stop the wheelchair and wait for SPACE or ENTER before continuing.
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

from angular_fuzzy_controller import AngularFuzzyController
from linear_fuzzy_controller import LinearFuzzyController


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


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
        # Topics / experiment.
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
        self.run_id = int(rospy.get_param("~run_id", 1))

        # Waypoint acceptance + state hysteresis.
        self.d_tolerance = float(rospy.get_param("~d_tolerance", 0.05))
        self.theta_enter = float(
            rospy.get_param("~theta_enter", math.radians(3.0))
        )
        self.theta_exit = float(
            rospy.get_param("~theta_exit", math.radians(7.0))
        )
        if self.d_tolerance <= 0.0:
            raise rospy.ROSException("d_tolerance must be > 0")
        if self.theta_enter <= 0.0:
            raise rospy.ROSException("theta_enter must be > 0")
        if self.theta_exit <= self.theta_enter:
            raise rospy.ROSException("theta_exit must be greater than theta_enter")

        # Physical command limits.  There are deliberately no minimum-speed
        # clamps; the fuzzy controllers own the nonlinear command mapping.
        self.max_linear_velocity = float(
            rospy.get_param("~max_linear_velocity", 0.25)
        )
        self.max_angular_velocity = float(
            rospy.get_param("~max_angular_velocity", 0.45)
        )
        self.linear_acceleration = float(
            rospy.get_param("~linear_acceleration", 0.12)
        )
        self.angular_acceleration = float(
            rospy.get_param("~angular_acceleration", 0.60)
        )
        if self.max_linear_velocity <= 0.0:
            raise rospy.ROSException("max_linear_velocity must be > 0")
        if self.max_angular_velocity <= 0.0:
            raise rospy.ROSException("max_angular_velocity must be > 0")
        if self.linear_acceleration <= 0.0:
            raise rospy.ROSException("linear_acceleration must be > 0")
        if self.angular_acceleration <= 0.0:
            raise rospy.ROSException("angular_acceleration must be > 0")

        self.control_hz = float(rospy.get_param("~control_hz", 20.0))
        self.pose_timeout = float(rospy.get_param("~pose_timeout", 0.50))

        # Fuzzy controllers live in their own modules.
        try:
            self.linear_controller = LinearFuzzyController(
                max_linear_velocity=self.max_linear_velocity
            )
            self.angular_controller = AngularFuzzyController(
                max_angular_velocity=self.max_angular_velocity
            )
        except Exception as exc:
            raise rospy.ROSException(
                "Could not initialise fuzzy controllers: {}. "
                "Install pyfuzzylite with: python3 -m pip install pyfuzzylite==8.0.6"
                .format(exc)
            )

        # Trajectory.  Normally overwritten by the YAML loaded in the launch.
        default_waypoints = [
            {"name": "P1", "x": 3.0, "y": 0.0, "checkpoint": True},
            {"name": "P2", "x": 3.0, "y": 3.5, "checkpoint": True},
            {"name": "P3", "x": 0.0, "y": 3.5, "checkpoint": True},
            {"name": "P4", "x": 0.0, "y": 0.0, "checkpoint": True},
        ]
        raw_waypoints = rospy.get_param("~waypoints", default_waypoints)
        if not raw_waypoints:
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

        # Logging.
        self.log_path = rospy.get_param(
            "~log_path", "/tmp/simple_planner_events.csv"
        )
        self.ground_truth_path = rospy.get_param(
            "~ground_truth_path", "/tmp/ground_truth.csv"
        )

        # Runtime state.
        self.pose = None
        self.last_pose_rx_time = None
        self.state = self.WAITING
        self.running = False
        self.target_index = 0
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_control_time = None
        self.prev_heading_error = None
        self.checkpoint_continue = threading.Event()
        self.keyboard_thread = None

        # ROS interfaces.
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.checkpoint_pub = rospy.Publisher(
            self.checkpoint_topic, String, queue_size=10
        )
        self.target_pub = rospy.Publisher(
            self.target_topic, Pose2D, queue_size=1, latch=True
        )
        self.pose_sub = rospy.Subscriber(
            self.pose_topic, Pose2D, self.pose_callback, queue_size=1
        )
        self.start_srv = rospy.Service("~start", Trigger, self.start_callback)
        self.stop_srv = rospy.Service("~stop", Trigger, self.stop_callback)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_hz), self.control_callback
        )

        self.prepare_log()
        self.prepare_ground_truth_log()
        self.start_keyboard_thread()
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("sara_simple_planner ready (fuzzy control)")
        rospy.loginfo("Pose feedback: %s", self.pose_topic)
        rospy.loginfo("Velocity output: %s", self.cmd_vel_topic)
        rospy.loginfo(
            "Hysteresis: enter FORWARD <= %.2f deg | return ORIENTING >= %.2f deg",
            math.degrees(self.theta_enter),
            math.degrees(self.theta_exit),
        )
        rospy.loginfo(
            "Limits: vmax=%.3f m/s | wmax=%.3f rad/s | ramps=(%.3f, %.3f)",
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.linear_acceleration,
            self.angular_acceleration,
        )
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

    # ------------------------------------------------------------------
    # ROS callbacks / services
    # ------------------------------------------------------------------

    def pose_callback(self, msg):
        self.pose = msg
        self.last_pose_rx_time = rospy.Time.now()
        if not self.running and self.state == self.WAITING:
            self.start_motion()

    def start_callback(self, _req):
        if self.pose is None:
            return TriggerResponse(
                success=False,
                message="No pose received yet on {}".format(self.pose_topic),
            )
        if self.running:
            return TriggerResponse(success=False, message="Planner is already running")
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

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def reset_experiment(self):
        self.target_index = 0
        self.state = self.WAITING
        self.checkpoint_continue.clear()
        self.reset_controller()

    def start_motion(self):
        self.running = True
        self.target_index = 0
        self.checkpoint_continue.clear()
        self.reset_controller()
        self.publish_target()
        self.select_motion_state()
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
            self.publish_zero(reset_history=True)
            return

        if self.state == self.FORWARD:
            self.control_forward(now)
        elif self.state == self.ORIENTING:
            self.control_orienting(now)
        elif self.state == self.CHECKPOINT:
            self.control_checkpoint()
        else:
            self.publish_zero()

    def target_errors(self):
        target = self.waypoints[self.target_index]
        dx = target["x"] - self.pose.x
        dy = target["y"] - self.pose.y
        distance_error = math.hypot(dx, dy)
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - self.pose.theta)
        return distance_error, heading_error

    def select_motion_state(self):
        """Choose the sensible state from the current heading error."""
        if self.pose is None:
            self.state = self.WAITING
            return
        _, heading_error = self.target_errors()
        self.state = (
            self.ORIENTING
            if abs(heading_error) >= self.theta_exit
            else self.FORWARD
        )
        self.prev_heading_error = None

    def control_forward(self, now):
        distance_error, heading_error = self.target_errors()
        dt = self.controller_dt(now)

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

        # True hysteresis: stay in FORWARD until the outer threshold is crossed.
        if abs(heading_error) >= self.theta_exit:
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
                    "State -> ORIENTING | heading error=%.2f deg",
                    math.degrees(heading_error),
                )
            return

        requested = self.linear_controller.compute(distance_error)
        linear = move_towards(
            self.last_linear_cmd,
            requested,
            self.linear_acceleration * dt,
        )
        self.prev_heading_error = None
        self.publish_cmd(linear, 0.0)

    def control_orienting(self, now):
        _distance_error, heading_error = self.target_errors()
        dt = self.controller_dt(now)

        # Inner threshold: stop rotating fully before resuming translation.
        if abs(heading_error) <= self.theta_enter:
            angular = move_towards(
                self.last_angular_cmd,
                0.0,
                self.angular_acceleration * dt,
            )
            self.publish_cmd(0.0, angular)
            if abs(angular) <= 1e-6:
                self.state = self.FORWARD
                self.prev_heading_error = None
                self.last_linear_cmd = 0.0
                rospy.loginfo(
                    "State -> FORWARD | heading error=%.2f deg",
                    math.degrees(heading_error),
                )
            return

        error_rate = 0.0
        if self.prev_heading_error is not None and dt > 1e-9:
            # Wrapped difference avoids a derivative spike at +/-pi.
            error_rate = wrap_angle(
                heading_error - self.prev_heading_error
            ) / dt

        requested = self.angular_controller.compute(
            heading_error,
            error_rate,
        )
        angular = move_towards(
            self.last_angular_cmd,
            requested,
            self.angular_acceleration * dt,
        )

        self.prev_heading_error = heading_error
        self.publish_cmd(0.0, angular)

    def waypoint_reached(self):
        target = self.waypoints[self.target_index]
        self.publish_zero(reset_history=True)
        if target["checkpoint"]:
            self.enter_checkpoint()
            return
        self.log_event("WAYPOINT_REACHED")
        rospy.loginfo("Waypoint %s reached (no checkpoint).", target["name"])
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
            "reference = (%.6f, %.6f)", target["x"], target["y"]
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
        self.reset_controller()
        self.publish_target()
        self.select_motion_state()
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

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def start_keyboard_thread(self):
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            name="simple_planner_keyboard",
            daemon=True,
        )
        self.keyboard_thread.start()

    def keyboard_loop(self):
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
                if char in (" ", "\n", "\r") and self.state == self.CHECKPOINT:
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

    # ------------------------------------------------------------------
    # Controller / command helpers
    # ------------------------------------------------------------------

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
        self.prev_heading_error = None
        self.linear_controller.reset()
        self.angular_controller.reset()

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)
        self.last_linear_cmd = linear
        self.last_angular_cmd = angular

    def publish_zero(self, reset_history=False):
        msg = Twist()
        self.cmd_pub.publish(msg)
        if reset_history:
            self.last_linear_cmd = 0.0
            self.last_angular_cmd = 0.0
            self.last_control_time = None
            self.prev_heading_error = None
            self.linear_controller.reset()
            self.angular_controller.reset()

    def publish_target(self):
        target = self.waypoints[self.target_index]
        msg = Pose2D()
        msg.x = target["x"]
        msg.y = target["y"]
        msg.theta = 0.0
        self.target_pub.publish(msg)
        rospy.loginfo(
            "New target: %s | x=%.3f y=%.3f | checkpoint=%s",
            target["name"], target["x"], target["y"], target["checkpoint"]
        )

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

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
                    "timestamp", "event", "state", "run_id", "waypoint",
                    "checkpoint", "ref_x", "ref_y", "pose_x", "pose_y",
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
                    "run_id", "timestamp", "checkpoint_id", "ref_x", "ref_y",
                    "gt_dx", "gt_dy", "gt_x", "gt_y",
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
                "", "", "", "",
            ]
        )
        self.gt_file.flush()

    def log_event(self, event, timestamp=None):
        if not self.waypoints:
            return
        target = self.waypoints[min(self.target_index, len(self.waypoints) - 1)]
        px = py = ptheta = float("nan")
        if self.pose is not None:
            px = self.pose.x
            py = self.pose.y
            ptheta = self.pose.theta
        if timestamp is None:
            timestamp = rospy.Time.now().to_sec()
        self.log_writer.writerow(
            [
                "{:.9f}".format(timestamp), event, self.state, self.run_id,
                target["name"], int(target["checkpoint"]), target["x"],
                target["y"], px, py, ptheta,
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
