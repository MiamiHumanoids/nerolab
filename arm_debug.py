# sudo ip link set can0 up type can bitrate 1000000
# If it doesn't stat up, do can down command replug in and candump

import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW


cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.V121,
    interface="socketcan",
    channel="can0",
)

robot = AgxArmFactory.create_arm(cfg)
robot.connect()

# Super Reset
# If getting RuntimeError: NERO Arm Joints did not re-enable after reset
# robot.set_follower_mode(); time.sleep(0.5); robot.reset(); time.sleep(1.0); print("Enabled:", robot.enable())

robot.set_auto_set_motion_mode_enabled(False)
robot.set_joint_limits_enabled(False)

print("connected:", robot.is_connected())
print("enable:", robot.enable())
robot.set_speed_percent(50)
print(robot.get_arm_status())

robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)

# pose_right= [0.8, 0.0, 0.0, -0.6, -0.6, 0.0, 0.0]
# pose_left= [0.8, 0.0, 0.0, 0.6, -0.6, 0.0, 0.0]
# pose_center = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# print("current angles:", robot.get_joint_angles())



print(robot.get_arm_status())

robot.move_j([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# status_after = robot.get_arm_status()
# print("status after tiny move:")
# print(status_after)

# angles_after = robot.get_joint_angles()
# if angles_after is not None:
#     print("angles after:", angles_after.msg)
# else:
#     print("angles after: None")

# print("\nIf motion_status stays REACH_TARGET_POS_FAILED, the robot is rejecting the target at the controller level.")



