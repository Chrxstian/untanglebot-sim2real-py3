# untanglebot_sim2real_py3

This project is the python3 part of the untanglebot-sim2real experiments. Since this project is only an addition to the `untanglebot_sim2real_py2` repository, make sure to check the other readme file for installation and setup instructions.

## Setup

Because the custom IK and vision libraries need python3 libraries, this repository uses `ROS Noetic`. Build the repository using `catkin_make` and `source devel/setup.bash`.

## Custom Velocity-based IK

For the pulling action we use a custom velocity-based controller to be able to stop the movement during the pulling action. You can start the ik_pull_service using `rosrun nextage_custom_ik ik_pull_service.py`

You can send movement commands using the service defined in `srv/ExecuteForcePull.srv`. The direction is in `WAIST` coordinate system. 

Because there is no collision avoidance using the velocity controller, I recommend to test the movement using the `dry_run` flag first. If the flag is set, the service will generate a `.gif` inside the visualization directory. If the movement looks good, you can test the same movement on the real robot.

```
rosservice call /execute_force_pull "{
  arm: 'left', 
  direction: [0.0, 0.0, -1.0], 
  force_threshold: 10.0,
  force_history_size: 10,
  dry_run: true
}"
```
