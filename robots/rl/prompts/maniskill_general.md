The base camera shows the workspace from a fixed external view. Use RGB-D segment or back_project to locate visible surfaces. When present, state.goal.xyz gives the environment's goal position in the same frame and units as move_to.

move_to targets the measured grasp center or pen tip. Omitting orientation_rpy preserves the measured wrist orientation. The controller checks position within 0.4 cm and orientation within 0.02 rad and allows up to 300 control steps per move, subject to the episode horizon. Both arms use the same mounting frame.
