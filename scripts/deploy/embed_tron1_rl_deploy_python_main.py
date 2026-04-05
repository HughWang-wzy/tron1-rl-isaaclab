import os
import sys
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import controllers as controllers


if __name__ == '__main__':
    robot_type = os.getenv("ROBOT_TYPE")
    if not robot_type:
        print("\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    rl_type = os.getenv("RL_TYPE")
    if not rl_type:
        print("\033[31mError: Please set the RL_TYPE using 'export RL_TYPE=isaacgym/isaaclab'.\033[0m")
        sys.exit(1)
    if rl_type != "isaacgym" and rl_type != "isaaclab":
        print("\033[31mError: RL_TYPE {} is not supported, choose between 'isaacgym' and 'isaaclab'.\033[0m".format(rl_type))
        sys.exit(1)

    robot = Robot(RobotType.PointFoot)

    robot_ip = "127.0.0.1"
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    if not robot.init(robot_ip):
        sys.exit()

    start_controller = robot_ip == "127.0.0.1"
    model_root = f'{os.path.dirname(os.path.abspath(__file__))}/controllers/model'

    if robot_type.startswith("PF"):
        controller = controllers.PointfootController(model_root, robot, robot_type, rl_type, start_controller)
        controller.run()
    elif robot_type.startswith("WF") and "MULTIEXPERT" in robot_type.upper():
        controller = controllers.WheelfootMultiExpertController(model_root, robot, robot_type, rl_type, start_controller)
        controller.run()
    elif robot_type.startswith("WF"):
        controller = controllers.WheelfootController(model_root, robot, robot_type, rl_type, start_controller)
        controller.run()
    elif robot_type.startswith("SF"):
        controller = controllers.SolefootController(model_root, robot, robot_type, rl_type, start_controller)
        controller.run()
    else:
        print(f"Error: unknow robot type '{robot_type}'")
