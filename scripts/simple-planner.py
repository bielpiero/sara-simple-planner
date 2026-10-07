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


class SimplePlanner:
    WAITING = "WAITING"
    GO_TO_POSITION = "GO_TO_POSITION"
    ALIGN_HEADING = "ALIGN_HEADING"
    CHECKPOINT = "CHECKPOINT"
    DONE = "DONE"
    STOPPED = "STOPPED"

    def __init__(self):
        # Topics
        self.pose_topic = rospy.get_param("~pose_topic", "/pose")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.checkpoint_topic = rospy.get_param(
            "~checkpoint_topic", "/simple_planner/checkpoint_reached"
        )
        self.target_topic = rospy.get_param(
            "~target_topic", "/simple_planner/target_pose"
        )

        # Controller
        self.kp_linear = rospy.get_param("~kp_linear", 0.45)
        self.kd_linear = rospy.get_param("~kd_linear", 0.08)
        self.kp_angular = rospy.get_param("~kp_angular", 1.30)
        self.kd_angular = rospy.get_param("~kd_angular", 0.10)

        self.position_tolerance = rospy.get_param("~position_tolerance", 0.10)
        self.heading_tolerance = rospy.get_param("~heading_tolerance", 0.10)

        self.max_linear_velocity = rospy.get_param(
            "~max_linear_velocity", 0.25
        )
        self.max_angular_velocity = rospy.get_param(
            "~max_angular_velocity", 0.45
        )

        # If the target is far away from the current heading, rotate first.
        self.rotate_in_place_threshold = rospy.get_param(
            "~rotate_in_place_threshold", 0.70
        )

        self.control_hz = rospy.get_param("~control_hz", 20.0)
        self.pose_timeout = rospy.get_param("~pose_timeout", 0.50)

        # Experiment
        self.loops = int(rospy.get_param("~loops", 5))
        self.dwell_time = rospy.get_param("~dwell_time", 2.0)
        self.manual_checkpoint = rospy.get_param("~manual_checkpoint", False)
        self.auto_start = rospy.get_param("~auto_start", False)

        # CSV event log
        self.log_path = rospy.get_param(
            "~log_path", "/tmp/simple_planner_events.csv"
        )

        raw_waypoints = rospy.get_param("~waypoints", [])
        if len(raw_waypoints) < 4:
            raise rospy.ROSException(
                "At least four waypoints are required in ~waypoints"
            )

        self.waypoints = []
        for i, wp in enumerate(raw_waypoints):
            self.waypoints.append(
                {
                    "name": str(wp.get("name", "P{}".format(i + 1))),
                    "marker_id": int(wp.get("marker_id", -1)),
                    "x": float(wp["x"]),
                    "y": float(wp["y"]),
                    "theta": float(wp.get("theta", 0.0)),
                }
            )

        self.waypoints.append(
            {
                "name": "start_point",
                "marker_id": -1,
                "x": 0.0,
                "y": 0.0,
                "theta": -1.57,
            }
        )

        # Runtime state
        self.pose = None
        self.last_pose_rx_time = None

        self.state = self.WAITING
        self.running = False

        # Initial target is P1 (home).
        self.target_index = 0
        self.current_loop = 0
        self.home_reached = False

        self.prev_distance_error = None
        self.prev_heading_error = None
        self.prev_control_time = None

        self.checkpoint_start_time = None
        self.waiting_manual_continue = False

        # ROS interfaces
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
                "{}(ArUco {})".format(w["name"], w["marker_id"])
                for w in self.waypoints
            ),
        )
        rospy.loginfo(
            "Experimental loops: %d, manual checkpoint: %s",
            self.loops,
            str(self.manual_checkpoint),
        )

        if self.auto_start:
            rospy.logwarn(
                "auto_start=true: motion will begin as soon as a valid pose is received"
            )

    # ------------------------------------------------------------
    # ROS callbacks / services
    # ------------------------------------------------------------

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
            return TriggerResponse(
                success=False, message="Planner is already running"
            )

        if self.state == self.DONE:
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
        self.publish_zero()
        self.log_event("STOPPED")

        return TriggerResponse(
            success=True, message="Planner stopped"
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
        self.current_loop = 0
        self.home_reached = False
        self.waiting_manual_continue = False
        self.checkpoint_start_time = None
        self.state = self.WAITING
        self.reset_pd()

    def start_motion(self):
        self.running = True
        self.state = self.GO_TO_POSITION
        self.reset_pd()
        self.publish_target()
        self.log_event("START")

    def control_callback(self, event):
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
            self.publish_zero()
            return

        if self.state == self.GO_TO_POSITION:
            self.control_position(now)
        elif self.state == self.ALIGN_HEADING:
            self.control_heading(now)
        elif self.state == self.CHECKPOINT:
            self.control_checkpoint(now)
        elif self.state in (self.DONE, self.STOPPED, self.WAITING):
            self.publish_zero()

    def control_position(self, now):
        target = self.waypoints[self.target_index]

        dx = target["x"] - self.pose.x
        dy = target["y"] - self.pose.y
        distance = math.hypot(dx, dy)

        if distance <= self.position_tolerance:
            self.publish_zero()
            self.state = self.ALIGN_HEADING
            self.reset_pd()
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - self.pose.theta)

        dt = self.controller_dt(now)

        d_distance = 0.0
        d_heading = 0.0

        if self.prev_distance_error is not None and dt > 0.0:
            d_distance = (
                distance - self.prev_distance_error
            ) / dt

        if self.prev_heading_error is not None and dt > 0.0:
            d_heading = wrap_angle(
                heading_error - self.prev_heading_error
            ) / dt

        linear = (
            self.kp_linear * distance
            + self.kd_linear * d_distance
        )

        angular = (
            self.kp_angular * heading_error
            + self.kd_angular * d_heading
        )

        # Do not advance while the robot is pointing far away from the path.
        if abs(heading_error) >= self.rotate_in_place_threshold:
            linear = 0.0
        else:
            # Smoothly reduce forward speed when not perfectly aligned.
            linear *= max(0.0, math.cos(heading_error))

        linear = clamp(
            linear, 0.0, self.max_linear_velocity
        )

        angular = clamp(
            angular,
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )

        self.publish_cmd(linear, angular)

        self.prev_distance_error = distance
        self.prev_heading_error = heading_error

    def control_heading(self, now):
        target = self.waypoints[self.target_index]

        heading_error = wrap_angle(
            target["theta"] - self.pose.theta
        )

        if abs(heading_error) <= self.heading_tolerance:
            self.publish_zero()
            self.enter_checkpoint()
            return

        dt = self.controller_dt(now)

        d_heading = 0.0
        if self.prev_heading_error is not None and dt > 0.0:
            d_heading = wrap_angle(
                heading_error - self.prev_heading_error
            ) / dt

        angular = (
            self.kp_angular * heading_error
            + self.kd_angular * d_heading
        )

        angular = clamp(
            angular,
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )

        self.publish_cmd(0.0, angular)
        self.prev_heading_error = heading_error

    def enter_checkpoint(self):
        self.state = self.CHECKPOINT
        self.checkpoint_start_time = rospy.Time.now()
        self.waiting_manual_continue = self.manual_checkpoint
        self.publish_zero()

        target = self.waypoints[self.target_index]

        event = (
            "loop={loop},waypoint={wp},marker_id={marker},"
            "ref_x={x:.8f},ref_y={y:.8f},ref_theta={theta:.8f},"
            "est_x={ex:.8f},est_y={ey:.8f},est_theta={eth:.8f}"
        ).format(
            loop=self.current_loop,
            wp=target["name"],
            marker=target["marker_id"],
            x=target["x"],
            y=target["y"],
            theta=target["theta"],
            ex=self.pose.x,
            ey=self.pose.y,
            eth=self.pose.theta,
        )

        self.checkpoint_pub.publish(String(data=event))
        self.log_event("CHECKPOINT")

        rospy.loginfo(
            "Checkpoint %s (ArUco %d) reached | loop=%d",
            target["name"],
            target["marker_id"],
            self.current_loop,
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
        self.reset_pd()

        # P1 is the home position. Reaching it initially only arms the route.
        if not self.home_reached:
            if self.target_index != 0:
                rospy.logerr("Internal planner error: home is not P1")
                self.running = False
                self.publish_zero()
                return

            self.home_reached = True
            self.current_loop = 1
            self.target_index = 1
            self.state = self.GO_TO_POSITION
            self.publish_target()

            rospy.loginfo(
                "Home reached. Starting loop %d/%d",
                self.current_loop,
                self.loops,
            )
            return

        # Normal polygon traversal: P2 -> P3 -> P4 -> P1
        if self.target_index < len(self.waypoints) - 1:
            self.target_index += 1
            self.state = self.GO_TO_POSITION
            self.publish_target()
            return

        # P4 reached: close the polygon by returning to P1.
        if self.target_index == len(self.waypoints) - 1:
            self.target_index = 0
            self.state = self.GO_TO_POSITION
            self.publish_target()
            return

        # P1 reached after P4 -> one complete loop.
        if self.target_index == 0:
            rospy.loginfo(
                "Loop %d/%d completed",
                self.current_loop,
                self.loops,
            )
            self.log_event("LOOP_COMPLETED")

            if self.current_loop >= self.loops:
                self.finish_experiment()
                return

            self.current_loop += 1
            self.target_index = 1
            self.state = self.GO_TO_POSITION
            self.publish_target()

            rospy.loginfo(
                "Starting loop %d/%d",
                self.current_loop,
                self.loops,
            )

    def finish_experiment(self):
        self.running = False
        self.state = self.DONE
        self.publish_zero()
        self.log_event("DONE")

        rospy.loginfo(
            "Experiment completed: %d polygon loops",
            self.loops,
        )

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def controller_dt(self, now):
        if self.prev_control_time is None:
            dt = 1.0 / self.control_hz
        else:
            dt = (now - self.prev_control_time).to_sec()
            if dt <= 0.0:
                dt = 1.0 / self.control_hz

        self.prev_control_time = now
        return dt

    def reset_pd(self):
        self.prev_distance_error = None
        self.prev_heading_error = None
        self.prev_control_time = None

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)

    def publish_zero(self):
        self.publish_cmd(0.0, 0.0)

    def publish_target(self):
        target = self.waypoints[self.target_index]

        msg = Pose2D()
        msg.x = target["x"]
        msg.y = target["y"]
        msg.theta = target["theta"]

        self.target_pub.publish(msg)

        rospy.loginfo(
            "New target: %s | ArUco %d | x=%.3f y=%.3f theta=%.3f",
            target["name"],
            target["marker_id"],
            target["x"],
            target["y"],
            target["theta"],
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
                    "loop",
                    "waypoint",
                    "marker_id",
                    "ref_x",
                    "ref_y",
                    "ref_theta",
                    "est_x",
                    "est_y",
                    "est_theta",
                ]
            )
            self.log_file.flush()

    def log_event(self, event):
        target = self.waypoints[self.target_index]

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
                self.current_loop,
                target["name"],
                target["marker_id"],
                target["x"],
                target["y"],
                target["theta"],
                px,
                py,
                ptheta,
            ]
        )
        self.log_file.flush()

    def shutdown(self):
        self.publish_zero()

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
