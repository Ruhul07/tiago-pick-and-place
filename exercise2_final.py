#!/usr/bin/env python3
"""
exercise2.py  (VERSION final-v1)

TIAGo moves a bottle and a cup from the middle table to the corner table.
Needs: Gazebo + MoveIt running, and perception.py running.

How it works
  1. Drive (ground-truth odometry + proportional controller) to the south side of the
     middle table, in front of the object.
  2. Head sweep: try several head tilts until perception.py reports a plausible pose.
  3. Measure the real gripper geometry from tf (where the finger centre is, which way
     the fingers point and close) so the grasp is orientation-correct for ANY URDF.
  4. MoveIt: come in above -> pre-grasp -> grasp, close fingers, check the object is
     really held (finger joint stops at the object width), lift, retreat.
  5. Drive around the middle table to the corner table, place, release, retreat.
Set PICK_ONLY=1 in the environment to stop after the first pick (test mode).
"""
import math
import os
import re
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Twist, PoseStamped, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             MotionPlanRequest, PlanningOptions, MoveItErrorCodes,
                             CollisionObject, PlanningScene)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros

VERSION = 'final-v1'

# ------------------------------ CONFIG ------------------------------------
WORLD_FILE = os.path.expanduser('~/tiago_ws/worlds/tiago_exercise2.world')
START_POSE = (-3.0, -3.0, 0.785)           # spawn pose (only used if ground truth odom is missing)
# Middle table: top 1.5 (x) x 0.9 (y), centred at the origin, top surface z = 0.75.
# Robot stands on the south side facing +y, right in front of each object.
STAND_MIDDLE = {'bottle': (-0.28, -0.85, math.pi / 2),
                'cup':    (0.25, -0.85, math.pi / 2)}
WAYPOINT = (1.6, -0.85, 0.0)               # clear of the middle table corner
STAND_CORNER = (3.0, 1.60, math.pi / 2)    # corner table is at (3, 2.5)
PLACE_X = 0.70                             # distance in front of base where objects are put down
PLACE_Y = {'bottle': -0.10, 'cup': 0.10}   # keep the two objects apart (base frame)
CMD_VEL = '/mobile_base_controller/cmd_vel_unstamped'
ODOM = '/mobile_base_controller/odom'
GT_ODOM = '/ground_truth_odom'
GRIPPER_ACTION = '/gripper_controller/follow_joint_trajectory'
HEAD_ACTION = '/head_controller/follow_joint_trajectory'
GRIPPER_OPEN = 0.044
HEAD_TILTS = [0.55, 0.35, 0.75, 0.15, -0.35, -0.55, -0.75]   # sign-agnostic sweep
LIFT = 0.10
# --------------------------------------------------------------------------


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_from_matrix(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def object_radius(model, default):
    """Read the object's radius (or half box width) from the world file."""
    try:
        w = open(WORLD_FILE).read()
        blk = re.search(r"<model name='%s'>(.*?)</model>" % model, w, re.S).group(1)
        c = re.search(r'<cylinder>\s*<radius>([\d.]+)</radius>', blk)
        if c:
            return float(c.group(1))
        b = re.search(r'<box>\s*<size>([\d.]+)\s+([\d.]+)', blk)
        if b:
            return min(float(b.group(1)), float(b.group(2))) / 2
    except Exception:
        pass
    return default


class PickPlace(Node):
    def __init__(self):
        super().__init__('exercise2')
        self.cmd = self.create_publisher(Twist, CMD_VEL, 10)
        self.odom = None
        self.gt = None
        self.js = {}
        self.poses = {}
        self.good_tilt = None
        self.create_subscription(Odometry, ODOM, lambda m: setattr(self, 'odom', m),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, GT_ODOM, lambda m: setattr(self, 'gt', m),
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_js, qos_profile_sensor_data)
        for n in ('bottle', 'cup'):
            self.create_subscription(
                PoseStamped, f'/perception/{n}_pose',
                lambda m, n=n: self.poses.__setitem__(n, (m, time.time())), 10)
        self.mg = ActionClient(self, MoveGroup, '/move_action')
        self.grip = ActionClient(self, FollowJointTrajectory, GRIPPER_ACTION)
        self.head = ActionClient(self, FollowJointTrajectory, HEAD_ACTION)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.R = np.eye(3)        # tool orientation (base frame) that points the fingers forward
        self.off = np.array([0.17, 0.0, 0.0])   # arm_tool_link -> finger centre (tool frame)
        self.R_used = np.eye(3)                 # orientation that MoveIt last succeeded with
        self.corr = np.zeros(3)                 # systematic arm tracking error (base frame)
        # "Grasp-fix": once the fingers have physically closed on an object, Gazebo is told to keep
        # the object glued to the hand (Gazebo-classic friction alone cannot hold it reliably).
        self.attached = None                      # (gazebo model name, offset to fingers in base frame)
        self.set_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.create_timer(0.03, self.follow_cb)
        self.radius = {'bottle': object_radius('blue_bottle', 0.03),
                       'cup': object_radius('red_cup', 0.04)}
        self.get_logger().info(f'script version {VERSION}; object radii (m): {self.radius}')

    # ---------------- basic helpers ----------------
    def on_js(self, m):
        for n, p in zip(m.name, m.position):
            self.js[n] = p

    def spin_for(self, sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def world_pose(self):
        """Robot pose in the world. Prefer Gazebo ground truth, else odom + spawn pose."""
        if self.gt is not None:
            g = self.gt.pose.pose
            return g.position.x, g.position.y, yaw_of(g.orientation)
        o = self.odom.pose.pose
        c, s = math.cos(START_POSE[2]), math.sin(START_POSE[2])
        return (START_POSE[0] + c * o.position.x - s * o.position.y,
                START_POSE[1] + s * o.position.x + c * o.position.y,
                START_POSE[2] + yaw_of(o.orientation))

    # ---------------- base motion ----------------
    def drive_to(self, gx, gy, gyaw):
        def stop():
            self.cmd.publish(Twist())

        def turn_to(target):
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                yaw = self.world_pose()[2]
                err = math.atan2(math.sin(target - yaw), math.cos(target - yaw))
                if abs(err) < 0.04:
                    break
                w = max(-0.6, min(0.6, 1.5 * err))
                if abs(w) < 0.25:                      # minimum speed so the base really turns
                    w = 0.25 if err > 0 else -0.25
                t = Twist()
                t.angular.z = w
                self.cmd.publish(t)
            stop()

        while self.odom is None and self.gt is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().warn('waiting for odometry ...', throttle_duration_sec=3.0)
        x, y, _ = self.world_pose()
        if math.hypot(gx - x, gy - y) > 0.1:
            turn_to(math.atan2(gy - y, gx - x))
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
                x, y, yaw = self.world_pose()
                dist = math.hypot(gx - x, gy - y)
                if dist < 0.05:
                    break
                bearing = math.atan2(gy - y, gx - x)
                err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))
                t = Twist()
                t.linear.x = min(0.4, 0.8 * dist)
                t.angular.z = 1.2 * err
                self.cmd.publish(t)
                self.get_logger().info(f'driving: at ({x:.2f},{y:.2f}) -> ({gx:.2f},{gy:.2f})',
                                       throttle_duration_sec=2.0)
            stop()
        turn_to(gyaw)
        self.spin_for(1.0)
        self.get_logger().info('base reached goal')

    # ---------------- head / gripper ----------------
    def _traj(self, client, joints, positions, sec):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('controller action server missing')
            return
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = joints
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = sec
        g.trajectory.points = [pt]
        fut = client.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res)

    def set_head(self, tilt):
        self._traj(self.head, ['head_1_joint', 'head_2_joint'], [0.0, tilt], 2)
        self.spin_for(1.0)

    def set_gripper(self, pos):
        self._traj(self.grip, ['gripper_left_finger_joint', 'gripper_right_finger_joint'],
                   [pos, pos], 2)
        self.spin_for(1.0)

    # ---------------- perception with automatic head tilt ----------------
    def see(self, name, wait=2.5):
        """Wait for a fresh, plausible pose of `name` (base_footprint frame)."""
        self.poses.pop(name, None)
        end = time.time() + wait
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if name in self.poses:
                msg, t = self.poses[name]
                p = msg.pose.position
                if time.time() - t < 1.0 and 0.75 < p.z < 0.95 and 0.4 < p.x < 1.3 and abs(p.y) < 0.4:
                    return p.x, p.y, p.z
        return None

    def acquire(self, name):
        """Find a head tilt from which the object is detected; remember it."""
        tilts = ([self.good_tilt] if self.good_tilt is not None else []) + HEAD_TILTS
        for tilt in tilts:
            self.set_head(tilt)
            p = self.see(name)
            if p:
                self.good_tilt = tilt
                self.get_logger().info(f'{name} seen with head tilt {tilt}: '
                                       f'x={p[0]:.3f} y={p[1]:.3f} z={p[2]:.3f}')
                return p
            self.get_logger().info(f'{name} not seen with head tilt {tilt}, trying next')
        return None

    # ---------------- gripper geometry from tf ----------------
    def measure_gripper(self):
        """Learn, from the robot's own tf, where the finger centre is and how the fingers point/close."""
        self.spin_for(2.0)
        try:
            def pos(child):
                t = self.tf_buffer.lookup_transform('arm_tool_link', child, Time())
                v = t.transform.translation
                return np.array([v.x, v.y, v.z])
            off = pos('gripper_grasping_frame')
            left, right = pos('gripper_left_finger_link'), pos('gripper_right_finger_link')
            d = off / np.linalg.norm(off)                  # direction the fingers point
            c = left - right
            c = c - c.dot(d) * d
            c = c / np.linalg.norm(c)                      # direction the fingers close along
            M = np.stack([d, c, np.cross(d, c)], axis=1)
            self.R = M.T      # tool orientation: fingers point along base x, close along base y
            self.off = off
            self.get_logger().info(f'gripper measured: finger centre {np.round(off, 3)} from arm_tool_link')
        except Exception as e:
            self.get_logger().warn(f'could not measure gripper from tf ({e}); using defaults')

    # ---------------- MoveIt ----------------
    def _plan(self, tool_xyz, R):
        self.mg.wait_for_server()
        pc = PositionConstraint()
        pc.header.frame_id = 'base_footprint'
        pc.link_name = 'arm_tool_link'
        pc.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.01, 0.01, 0.01]))
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, tool_xyz)
        p.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(p)
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = 'base_footprint'
        oc.link_name = 'arm_tool_link'
        qx, qy, qz, qw = quat_from_matrix(R)
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = qx, qy, qz, qw
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.12
        oc.weight = 1.0
        req = MotionPlanRequest()
        req.group_name = 'arm_torso'
        req.num_planning_attempts = 8
        req.allowed_planning_time = 8.0
        req.max_velocity_scaling_factor = 0.15
        req.max_acceleration_scaling_factor = 0.15
        req.goal_constraints = [Constraints(position_constraints=[pc], orientation_constraints=[oc])]
        goal = MoveGroup.Goal(request=req, planning_options=PlanningOptions(plan_only=False))
        fut = self.mg.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle.accepted:
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        return res.result().result.error_code.val == MoveItErrorCodes.SUCCESS

    def move_grasp(self, x, y, z):
        """Move so the CENTRE OF THE FINGERS is at (x, y, z) in base_footprint, fingers pointing forward."""
        Rf = np.diag([1.0, -1.0, -1.0]) @ self.R          # same direction, hand turned 180 deg
        for R in (self.R, Rf):
            tool = np.array([x, y, z]) - R @ self.off     # where arm_tool_link must be
            if self._plan(tool, R):
                self.R_used = R
                self.spin_for(0.5)
                self.get_logger().info(f'arm: fingers -> ({x:.2f},{y:.2f},{z:.2f}) OK')
                return True
        self.get_logger().error(f'arm: fingers -> ({x:.2f},{y:.2f},{z:.2f}) FAILED')
        return False

    GAZEBO_NAME = {'bottle': 'blue_bottle', 'cup': 'red_cup'}

    def fingers_tf(self):
        """Finger centre in base_footprint from the latest tf (no spinning, safe inside a timer)."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_footprint', 'gripper_grasping_frame', Time()).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:
            return None

    def follow_cb(self):
        """Every 30 ms: put the attached object where the hand is (upright, at its grasp offset)."""
        if self.attached is None or not self.set_state.service_is_ready():
            return
        name, rel = self.attached
        g = self.fingers_tf()
        if g is None or (self.gt is None and self.odom is None):
            return
        rx, ry, ryaw = self.world_pose()
        b = g + rel                                   # object position in the base frame
        c, s = math.cos(ryaw), math.sin(ryaw)
        st = EntityState()
        st.name = name
        st.reference_frame = 'world'
        st.pose.position.x = rx + c * b[0] - s * b[1]
        st.pose.position.y = ry + s * b[0] + c * b[1]
        st.pose.position.z = float(b[2])
        st.pose.orientation.w = 1.0                   # keep it upright
        self.set_state.call_async(SetEntityState.Request(state=st))

    def attach(self, name, rel):
        if not self.set_state.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/gazebo/set_entity_state missing: object is only held by '
                                   'friction (add the gazebo_ros_state plugin to the world)')
            return
        rel = np.array(rel, dtype=float)
        if np.linalg.norm(rel) > 0.06:
            rel = np.zeros(3)
        self.attached = (self.GAZEBO_NAME[name], rel)
        self.get_logger().info(f'GRASP-FIX: {name} attached to the gripper (offset {np.round(rel, 3)})')

    def detach(self):
        if self.attached is not None:
            self.get_logger().info(f'GRASP-FIX: {self.attached[0]} released')
        self.attached = None

    def cartesian(self, pts, slow=3.0):
        """Move the finger centre in a STRAIGHT LINE through pts (list of (x, y, z)).
        Unlike a joint-space plan, the hand cannot swing sideways and knock the object over."""
        cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('/compute_cartesian_path not available')
            return False
        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_footprint'
        req.start_state.is_diff = True               # start from the current arm state
        req.group_name = 'arm_torso'
        req.link_name = 'arm_tool_link'
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        qx, qy, qz, qw = quat_from_matrix(self.R_used)
        for (x, y, z) in pts:
            tool = np.array([x, y, z]) - self.corr - self.R_used @ self.off
            p = Pose()
            p.position.x, p.position.y, p.position.z = map(float, tool)
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
            req.waypoints.append(p)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res.fraction < 0.95:
            self.get_logger().warn(f'straight-line path only {res.fraction:.0%} feasible')
            return False
        traj = res.solution
        for pt in traj.joint_trajectory.points:       # slow it down (gentle near the object)
            t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) * slow
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            pt.velocities = [v / slow for v in pt.velocities]
            pt.accelerations = [a / slow ** 2 for a in pt.accelerations]
        ac = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        if not ac.wait_for_server(timeout_sec=5.0):
            return False
        gf = ac.send_goal_async(ExecuteTrajectory.Goal(trajectory=traj))
        rclpy.spin_until_future_complete(self, gf)
        rf = gf.result().get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        ok = rf.result().result.error_code.val == MoveItErrorCodes.SUCCESS
        self.spin_for(0.5)
        self.get_logger().info(f'straight line -> {np.round(pts[-1], 3)} {"OK" if ok else "FAILED"}')
        return ok

    def fingers_now(self):
        """Where the finger centre REALLY is (from tf), in base_footprint."""
        self.spin_for(1.0)
        try:
            t = self.tf_buffer.lookup_transform(
                'base_footprint', 'gripper_grasping_frame', Time()).transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception as e:
            self.get_logger().warn(f'fingers_now failed: {e}')
            return None

    def move_grasp_precise(self, x, y, z, tries=2):
        """Closed loop: command, read the real finger position, cancel the error, repeat."""
        ok = self.move_grasp(x, y, z)
        if not ok:
            return False
        for i in range(tries):
            a = self.fingers_now()
            if a is None:
                break
            err = a - np.array([x, y, z])
            if i == 0:
                self.corr = err.copy()          # remember the arm's systematic offset
            self.get_logger().info(f'DIAG finger error (x,y,z) = {np.round(err * 100, 1)} cm '
                                   f'(target {x:.3f},{y:.3f},{z:.3f})')
            if np.linalg.norm(err) < 0.008:
                break
            ok = self.move_grasp(x - err[0], y - err[1], z - err[2])
            if not ok:
                break
        return ok

    def _table_op(self, op):
        cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        if not cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('/apply_planning_scene not available, planning without the table')
            return
        rx, ry, ryaw = self.world_pose()
        dx, dy = -rx, -ry                                 # table centre = world origin
        c, s = math.cos(ryaw), math.sin(ryaw)
        co = CollisionObject()
        co.header.frame_id = 'base_footprint'
        co.id = 'center_table'
        co.operation = op
        if op == CollisionObject.ADD:
            pose = Pose()
            pose.position.x = c * dx + s * dy
            pose.position.y = -s * dx + c * dy
            pose.position.z = 0.70
            pose.orientation.z = math.sin(-ryaw / 2)
            pose.orientation.w = math.cos(-ryaw / 2)
            co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[1.5, 0.9, 0.1])]
            co.primitive_poses = [pose]
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects = [co]
        fut = cli.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, fut)

    # ---------------- pick and place ----------------
    def pick(self, name):
        target = max(self.radius[name] - 0.008, 0.004)    # fingers squeeze slightly
        for attempt in (1, 2):
            self.get_logger().info(f'--- pick {name}, attempt {attempt} ---')
            p = self.acquire(name)
            if p is None:
                self.get_logger().error(f'{name} never detected (check perception.py)')
                continue
            x, y, z = p
            self.set_gripper(GRIPPER_OPEN)
            self.get_logger().info(
                f'DIAG gripper joints L={self.js.get("gripper_left_finger_joint", -1):.3f} '
                f'R={self.js.get("gripper_right_finger_joint", -1):.3f} (open should be ~0.044)')
            self._table_op(CollisionObject.ADD)
            self.corr = np.zeros(3)
            ok = (self.move_grasp_precise(x - 0.15, y, z + 0.12)   # above and well in front (joint space)
                  and self.cartesian([(x - 0.15, y, z)])           # straight DOWN, still far from the object
                  and self.cartesian([(x, y, z)]))                 # straight IN: fingers around the object
            if not ok:
                self._table_op(CollisionObject.REMOVE)
                self.move_grasp(x - 0.15, y, z + 0.12)
                continue
            self.set_gripper(target)
            self.spin_for(1.0)
            f = self.js.get('gripper_left_finger_joint', 0.0)
            fnow = self.fingers_now()
            near = fnow is not None and np.linalg.norm(fnow - np.array([x, y, z])) < 0.035
            # contact test: fingers stopped on the object. FORCE_ATTACH=1 also accepts "fingers are at
            # the object's perceived position" (use only if contact keeps being missed).
            held = (f > target + 0.004) or bool(os.environ.get('FORCE_ATTACH') and near)
            held_dummy = f > target + 0.004                         # fingers stopped on the object
            self.get_logger().info(f'finger joint {f:.3f} (empty would be {target:.3f}) -> '
                                   f'{"HELD" if held else "MISSED"}')
            if held:
                self.attach(name, (np.array([x, y, z]) - fnow) if fnow is not None else np.zeros(3))
                if not self.cartesian([(x, y, z + LIFT)]):            # straight UP
                    self.move_grasp(x, y, z + LIFT)
                if not self.cartesian([(x - 0.10, y, z + LIFT)]):     # straight BACK
                    self.move_grasp(x - 0.10, y, z + LIFT)
                self._table_op(CollisionObject.REMOVE)
                return z
            self.set_gripper(GRIPPER_OPEN)                    # missed: open, back off, retry
            self.move_grasp(x - 0.15, y, z + 0.12)
            self._table_op(CollisionObject.REMOVE)
        return None

    def place(self, name, z):
        py = PLACE_Y[name]
        self.move_grasp(PLACE_X - 0.10, py, z + LIFT + 0.05)
        self.move_grasp(PLACE_X, py, z + LIFT + 0.05)
        self.move_grasp(PLACE_X, py, z + 0.03)
        self.detach()                                         # let go of the object (then open)                # just above the table top
        self.set_gripper(GRIPPER_OPEN)
        self.move_grasp(PLACE_X - 0.12, py, z + 0.03)         # slide the fingers back out
        self.move_grasp(PLACE_X - 0.12, py, z + LIFT + 0.10)

    def carry_pose(self):
        self.move_grasp(0.45, 0.0, 1.0)

    def run(self):
        self.measure_gripper()
        for name in ('bottle', 'cup'):
            self.get_logger().info(f'=== moving {name} ===')
            self.drive_to(*STAND_MIDDLE[name])
            z = self.pick(name)
            if z is None:
                self.get_logger().error(f'{name}: pick FAILED, skipping')
                continue
            if os.environ.get('PICK_ONLY'):
                self.get_logger().info(f'PICK_ONLY: {name} is in the gripper, stopping')
                return
            self.carry_pose()
            self.drive_to(*WAYPOINT)
            self.drive_to(*STAND_CORNER)
            self.place(name, z)
            self.carry_pose()
            self.drive_to(*WAYPOINT)
            self.get_logger().info(f'{name}: DONE')
        self.get_logger().info('=== all objects processed ===')


def main():
    rclpy.init()
    node = PickPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        node.cmd.publish(Twist())
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
