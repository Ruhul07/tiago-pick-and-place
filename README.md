# tiago-pick-and-place
Autonomous pick-and-place with the TIAGo robot in Gazebo using ROS 2 Humble, MoveIt 2 and RGB-D perception.
cd ~/tiago_ws
cat > README.md << 'EOF'
# TIAGo Pick-and-Place (ROS 2 Humble + MoveIt 2)

Autonomous mobile manipulation in Gazebo: a TIAGo robot detects a bottle and a cup on a middle table and moves both to a corner table.

**Demo video:** PASTE_VIDEO_LINK

## Features
- **Perception:** RGB-D colour segmentation + depth back-projection gives 6DOF object poses (`scripts/perception.py`)
- **Navigation:** proportional base controller using ground-truth odometry
- **Manipulation:** MoveIt 2 (`arm_torso` group) with Cartesian approach, lift and place
- **Orchestration:** one rclpy node runs the whole task (`scripts/exercise2.py`)

## Tech stack
Ubuntu 22.04, ROS 2 Humble, Gazebo Classic, MoveIt 2, PAL TIAGo public simulation, Python (rclpy, tf2, NumPy)

## Repository layout
- `worlds/` Gazebo world files (tables, bottle, cup)
- `scripts/exercise1.py` drive to the table and move the arm above it
- `scripts/perception.py` detects bottle and cup, publishes `/perception/{bottle,cup}_pose`
- `scripts/exercise2.py` full pick-and-place

## How to run
1. Terminal 1: launch Gazebo + MoveIt with `tiago_exercise2.world` (`tiago_gazebo` with `moveit:=True`)
2. Terminal 2: `python3 scripts/perception.py`
3. Terminal 3: `python3 scripts/exercise2.py`

## Limitations (honest note)
Contact-based grasping in Gazebo Classic is unreliable. The grasp and final set-down are therefore simulated with the Gazebo state plugin (`/gazebo/set_entity_state`). Perception, navigation and arm planning run as they would on the real robot.

## Author
Your Name, [LinkedIn link]
EOF

git add README.md
git commit -m "Update README"
git push
