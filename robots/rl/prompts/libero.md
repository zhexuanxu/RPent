Robot: one Franka arm with a parallel-jaw gripper. Positions are absolute world-frame end-effector coordinates in metres. target_yaw and target_pitch follow move_to's definitions, in radians. opening is the normalized jaw gap, 0 closed and 1 fully open.

The robot stands at the -x edge facing +x; +y is its left and +z is up. agentview looks back toward the robot from the far side; wrist moves with the gripper. Motion follows a straight line without collision avoidance. Visible surface heights describe the observed top or side of an object rather than a gripper target.

The pose reference is between the finger pads, about 9.7 cm beyond the wrist flange. Fingertips extend about 0.9 cm below this point. Fully open, the jaws span about 7.8 cm. They close along the gripper y-axis, perpendicular to its yaw heading; yaw 0 closes along world y. These are robot dimensions, not object-centre or grasp estimates.
