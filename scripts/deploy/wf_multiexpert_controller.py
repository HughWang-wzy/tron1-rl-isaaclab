"""Standalone wheelfoot multi-expert deploy controller.

This controller mirrors the original Tron1 wheelfoot deploy pipeline and adds
the extra inputs required by ``WFMultiExpertFlatEnvCfg``:

1. ``commands`` (shared base velocity command)
2. ``jump_commands`` (jump trigger, target height, standing height)
3. ``gait_commands`` (frequency, offset, duration, swing height, standing height)
4. ``env_group`` (0 = jump expert, 1 = gait expert)

It supports both exported layouts:

1. ``encoder.onnx`` + ``policy.onnx`` for the old deploy flow
2. ``student_policy.onnx`` for a single fused student model
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from functools import partial

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

import limxsdk.datatypes as datatypes
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType


class WheelfootMultiExpertController:
    def __init__(
        self,
        model_dir: str,
        robot,
        robot_type: str,
        rl_type: str | None = None,
        start_controller: bool = False,
        config_path: str | None = None,
    ):
        self.robot = robot
        self.robot_type = robot_type
        self.rl_type = rl_type
        self.model_dir = os.path.abspath(model_dir)

        model_root = self.model_dir
        if config_path is not None:
            self.config_file = os.path.abspath(config_path)
            artifact_dir = os.path.dirname(self.config_file)
        elif os.path.isdir(os.path.join(self.model_dir, self.robot_type)):
            model_root = os.path.join(self.model_dir, self.robot_type)
            self.config_file = os.path.join(model_root, "params.yaml")
            policy_root = os.path.join(model_root, "policy")
            if self.rl_type is None:
                rl_candidates = [name for name in os.listdir(policy_root)] if os.path.isdir(policy_root) else []
                self.rl_type = "isaaclab" if "isaaclab" in rl_candidates else (rl_candidates[0] if rl_candidates else "isaaclab")
            artifact_dir = os.path.join(policy_root, self.rl_type)
        else:
            self.config_file = os.path.join(self.model_dir, "params.yaml")
            artifact_dir = self.model_dir

        self.deploy_meta_file = os.path.join(artifact_dir, "deploy_meta.json")
        self.model_policy = os.path.join(artifact_dir, "policy.onnx")
        self.model_encoder = os.path.join(artifact_dir, "encoder.onnx")
        self.model_student = os.path.join(artifact_dir, "student_policy.onnx")

        self.load_config(self.config_file)
        self.load_deploy_meta(self.deploy_meta_file)
        self.initialize_onnx_models()

        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.q = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.dq = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.tau = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg["stiffness"] for _ in range(self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg["damping"] for _ in range(self.joint_num)]

        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0.0 for _ in range(self.joint_num)]
        self.robot_state.q = [0.0 for _ in range(self.joint_num)]
        self.robot_state.dq = [0.0 for _ in range(self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0.0
        self.imu_data.quat[1] = 0.0
        self.imu_data.quat[2] = 0.0
        self.imu_data.quat[3] = 1.0
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        self.calibration_state = -1
        self.start_controller = start_controller
        self.is_first_rec_obs = True
        self.loop_count = 0
        self.stand_percent = 0.0
        self.mode = "STAND"

        self.prev_buttons: list[int] = []
        self.manual_expert_mode = self.expert_cfg["default_mode"]
        self.jump_active_until = 0.0

    def load_config(self, config_file: str) -> None:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cfg = config.get("WFMultiExpertCfg", config.get("PointfootCfg"))
        if cfg is None:
            raise KeyError("params.yaml must contain 'WFMultiExpertCfg' or 'PointfootCfg'.")

        self.joint_names = cfg["joint_names"]
        self.init_state = cfg["init_state"]["default_joint_angle"]
        self.stand_duration = cfg["stand_mode"]["stand_duration"]
        self.control_cfg = cfg["control"]
        self.rl_cfg = cfg["normalization"]
        self.obs_scales = self.rl_cfg["obs_scales"]
        self.imu_orientation_offset = np.array(list(cfg["imu_orientation_offset"].values()), dtype=np.float64)
        self.user_cmd_cfg = cfg["user_cmd_scales"]
        self.loop_frequency = cfg["loop_frequency"]

        size_cfg = cfg["size"]
        self.actions_size = size_cfg["actions_size"]
        self.commands_size = size_cfg["commands_size"]
        self.observations_size = size_cfg["observations_size"]
        self.obs_history_length = size_cfg["obs_history_length"]
        self.encoder_output_size = size_cfg["encoder_output_size"]
        self.joint_pos_idxs = size_cfg["jointpos_idxs"]
        self.jump_commands_size = size_cfg.get("jump_commands_size", 3)
        self.gait_commands_size = size_cfg.get("gait_commands_size", 5)
        self.env_group_size = size_cfg.get("env_group_size", 1)

        self.joint_num = len(self.joint_names)
        self.wheel_joint_damping = self.control_cfg["wheel_joint_damping"]
        self.wheel_joint_torque_limit = self.control_cfg["wheel_joint_torque_limit"]

        self.init_joint_angles = np.zeros(self.joint_num, dtype=np.float64)
        for i, joint_name in enumerate(self.joint_names):
            self.init_joint_angles[i] = self.init_state[joint_name]

        self.proprio_history_buffer = np.zeros(self.obs_history_length * self.observations_size, dtype=np.float32)
        self.encoder_out = np.zeros(self.encoder_output_size, dtype=np.float32)
        self.actions = np.zeros(self.actions_size, dtype=np.float32)
        self.last_actions = np.zeros(self.actions_size, dtype=np.float32)
        self.observations = np.zeros(self.observations_size, dtype=np.float32)
        self.commands = np.zeros(self.commands_size, dtype=np.float32)
        self.scaled_commands = np.zeros(self.commands_size, dtype=np.float32)

        self.expert_cfg = {
            "default_mode": cfg.get("expert", {}).get("default_mode", "jump"),
            "auto_return_to_default": bool(cfg.get("expert", {}).get("auto_return_to_default", True)),
            "gait_env_group": float(cfg.get("expert", {}).get("gait_env_group", 1.0)),
            "jump_env_group": float(cfg.get("expert", {}).get("jump_env_group", 0.0)),
        }
        self.gait_cfg = {
            "frequency": float(cfg.get("gait", {}).get("frequency", 2.0)),
            "offset": float(cfg.get("gait", {}).get("offset", 0.5)),
            "duration": float(cfg.get("gait", {}).get("duration", 0.5)),
            "swing_height": float(cfg.get("gait", {}).get("swing_height", 0.15)),
            "standing_height": float(cfg.get("gait", {}).get("standing_height", 0.8)),
        }
        self.jump_cfg = {
            "trigger_button_index": int(cfg.get("jump", {}).get("trigger_button_index", 0)),
            "expert_toggle_button_index": cfg.get("jump", {}).get("expert_toggle_button_index"),
            "hold_seconds": float(cfg.get("jump", {}).get("hold_seconds", 1.0)),
            "standing_height": float(cfg.get("jump", {}).get("standing_height", self.gait_cfg["standing_height"])),
            "delta_height": float(cfg.get("jump", {}).get("delta_height", 0.35)),
        }
        if self.jump_cfg["expert_toggle_button_index"] is not None:
            self.jump_cfg["expert_toggle_button_index"] = int(self.jump_cfg["expert_toggle_button_index"])

        self.model_cfg = {
            "use_single_onnx": bool(cfg.get("model", {}).get("use_single_onnx", False)),
        }

    def load_deploy_meta(self, deploy_meta_file: str) -> None:
        self.deploy_meta = None
        if not os.path.isfile(deploy_meta_file):
            return
        with open(deploy_meta_file, "r", encoding="utf-8") as f:
            self.deploy_meta = json.load(f)

        dims = self.deploy_meta.get("dims", {})
        if dims.get("policy_dim", self.observations_size) != self.observations_size:
            raise ValueError(
                f"Observation size mismatch: params.yaml has {self.observations_size}, deploy_meta.json has {dims.get('policy_dim')}."
            )
        if dims.get("commands_dim", self.commands_size) != self.commands_size:
            raise ValueError(
                f"Commands size mismatch: params.yaml has {self.commands_size}, deploy_meta.json has {dims.get('commands_dim')}."
            )
        if dims.get("jump_commands_dim", self.jump_commands_size) != self.jump_commands_size:
            raise ValueError(
                "Jump command size mismatch between params.yaml and deploy_meta.json."
            )
        if dims.get("gait_commands_dim", self.gait_commands_size) != self.gait_commands_size:
            raise ValueError(
                "Gait command size mismatch between params.yaml and deploy_meta.json."
            )
        if dims.get("env_group_dim", self.env_group_size) != self.env_group_size:
            raise ValueError(
                "Env-group size mismatch between params.yaml and deploy_meta.json."
            )

    def initialize_onnx_models(self) -> None:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = False
        session_options.enable_mem_pattern = False

        providers = ["CPUExecutionProvider"]

        self.use_single_onnx = self.model_cfg["use_single_onnx"]
        if self.use_single_onnx and not os.path.isfile(self.model_student):
            raise FileNotFoundError(f"Configured single-model deploy, but file not found: {self.model_student}")
        if not self.use_single_onnx and (not os.path.isfile(self.model_policy) or not os.path.isfile(self.model_encoder)):
            if os.path.isfile(self.model_student):
                self.use_single_onnx = True
            else:
                raise FileNotFoundError(
                    "Expected either encoder.onnx + policy.onnx or student_policy.onnx in the exported model directory."
                )

        if self.use_single_onnx:
            self.student_session = ort.InferenceSession(
                self.model_student, sess_options=session_options, providers=providers
            )
            self.student_input_name = self.student_session.get_inputs()[0].name
            self.student_output_names = [item.name for item in self.student_session.get_outputs()]
            self.policy_session = None
            self.encoder_session = None
        else:
            self.policy_session = ort.InferenceSession(
                self.model_policy, sess_options=session_options, providers=providers
            )
            self.policy_input_name = self.policy_session.get_inputs()[0].name
            self.policy_output_names = [item.name for item in self.policy_session.get_outputs()]

            self.encoder_session = ort.InferenceSession(
                self.model_encoder, sess_options=session_options, providers=providers
            )
            self.encoder_input_name = self.encoder_session.get_inputs()[0].name
            self.encoder_output_names = [item.name for item in self.encoder_session.get_outputs()]
            self.student_session = None

    def run(self) -> None:
        while not self.start_controller:
            time.sleep(1)

        self.default_joint_angles = np.array([0.0] * self.joint_num, dtype=np.float64)
        self.stand_percent += 1.0 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        rate = Rate(self.loop_frequency)
        while self.start_controller:
            self.update()
            rate.sleep()

        self.robot_cmd.q = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.dq = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.tau = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.Kp = [0.0 for _ in range(self.joint_num)]
        self.robot_cmd.Kd = [1.0 for _ in range(self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    def handle_stand_mode(self) -> None:
        self.init_state["hip_L_Joint"] = -0.9
        self.init_state["hip_R_Joint"] = 0.9
        if self.stand_percent < 1.0:
            for joint_index in range(self.joint_num):
                if (joint_index + 1) % 4 != 0:
                    pos_des = (
                        self.default_joint_angles[joint_index] * (1.0 - self.stand_percent)
                        + self.init_state[self.joint_names[joint_index]] * self.stand_percent
                    )
                    self.set_joint_command(
                        joint_index,
                        pos_des,
                        0.0,
                        0.0,
                        self.control_cfg["stiffness"],
                        self.control_cfg["damping"],
                    )
                else:
                    self.set_joint_command(joint_index, 0.0, 0.0, 0.0, 0.0, self.wheel_joint_damping)
            self.stand_percent += 3.0 / (self.stand_duration * self.loop_frequency)
        else:
            self.mode = "WALK"

    def handle_walk_mode(self) -> None:
        self.init_joint_angles[1] = 0.0
        self.init_joint_angles[5] = 0.0

        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        if self.loop_count % self.control_cfg["decimation"] == 0:
            self.compute_observation()
            if not self.use_single_onnx:
                self.compute_encoder()
            self.compute_actions()
            clip_actions = self.rl_cfg["clip_scales"]["clip_actions"]
            self.actions = np.clip(self.actions, -clip_actions, clip_actions)
            self.actions = self.swap_positions(self.actions, reverse=True)

        joint_pos = np.array(self.robot_state_tmp.q, dtype=np.float64)
        joint_vel = np.array(self.robot_state_tmp.dq, dtype=np.float64)

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:
                action_min = (
                    joint_pos[i]
                    - self.init_joint_angles[i]
                    + (self.control_cfg["damping"] * joint_vel[i] - self.control_cfg["user_torque_limit"])
                    / self.control_cfg["stiffness"]
                )
                action_max = (
                    joint_pos[i]
                    - self.init_joint_angles[i]
                    + (self.control_cfg["damping"] * joint_vel[i] + self.control_cfg["user_torque_limit"])
                    / self.control_cfg["stiffness"]
                )
                self.actions[i] = max(
                    action_min / self.control_cfg["action_scale_pos"],
                    min(action_max / self.control_cfg["action_scale_pos"], self.actions[i]),
                )
                pos_des = self.actions[i] * self.control_cfg["action_scale_pos"] + self.init_joint_angles[i]
                self.set_joint_command(
                    i,
                    pos_des,
                    0.0,
                    0.0,
                    self.control_cfg["stiffness"],
                    self.control_cfg["damping"],
                )
                self.last_actions[i] = self.actions[i]
            else:
                action_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                action_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                self.last_actions[i] = self.actions[i]
                self.actions[i] = max(
                    action_min / self.wheel_joint_damping,
                    min(action_max / self.wheel_joint_damping, self.actions[i]),
                )
                velocity_des = self.actions[i] * self.wheel_joint_damping
                self.set_joint_command(i, 0.0, velocity_des, 0.0, 0.0, self.wheel_joint_damping)

    def swap_positions(self, initial_array, reverse: bool = False, exclude_wheel: bool = False):
        joint_idx_lab = [0, 3, 1, 4, 2, 5] if exclude_wheel else [0, 4, 1, 5, 2, 6, 3, 7]
        new_array = np.zeros(initial_array.shape, dtype=np.float64)
        for i, lab_index in enumerate(joint_idx_lab):
            if not reverse:
                new_array[i] = initial_array[lab_index]
            else:
                new_array[lab_index] = initial_array[i]
        return new_array

    def compute_observation(self) -> None:
        imu_orientation = np.array(self.imu_data_tmp.quat, dtype=np.float64)
        q_wi = R.from_quat(imu_orientation).as_euler("zyx")
        inverse_rot = R.from_euler("zyx", q_wi).inv().as_matrix()

        gravity_vector = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        projected_gravity = np.dot(inverse_rot, gravity_vector)

        base_ang_vel = np.array(self.imu_data_tmp.gyro, dtype=np.float64)
        rot = R.from_euler("zyx", self.imu_orientation_offset).as_matrix()
        base_ang_vel = np.dot(rot, base_ang_vel)
        projected_gravity = np.dot(rot, projected_gravity)

        joint_positions = np.array(self.robot_state_tmp.q, dtype=np.float64)
        joint_velocities = np.array(self.robot_state_tmp.dq, dtype=np.float64)
        actions = np.array(self.last_actions, dtype=np.float64)

        command_scaler = np.diag(
            [
                self.user_cmd_cfg["lin_vel_x"],
                self.user_cmd_cfg["lin_vel_y"],
                self.user_cmd_cfg["ang_vel_yaw"],
            ]
        )
        self.scaled_commands = np.dot(command_scaler, self.commands).astype(np.float32)

        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales["dof_pos"]
        joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs], dtype=np.float64)
        joint_pos_input = self.swap_positions(joint_pos_input, exclude_wheel=True)
        joint_velocities = self.swap_positions(joint_velocities)
        actions = self.swap_positions(actions)

        obs = np.concatenate(
            [
                base_ang_vel * self.obs_scales["ang_vel"],
                projected_gravity,
                joint_pos_input,
                joint_velocities * self.obs_scales["dof_vel"],
                actions,
            ],
            axis=0,
        ).astype(np.float32)

        if self.is_first_rec_obs:
            for i in range(self.obs_history_length):
                start = i * self.observations_size
                end = (i + 1) * self.observations_size
                self.proprio_history_buffer[start:end] = obs
            self.is_first_rec_obs = False

        self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size :]
        self.proprio_history_buffer[-self.observations_size :] = obs
        clip_obs = self.rl_cfg["clip_scales"]["clip_observations"]
        self.observations = np.clip(obs, -clip_obs, clip_obs).astype(np.float32)

    def compute_encoder(self) -> None:
        input_tensor = self.proprio_history_buffer.astype(np.float32)
        inputs = {self.encoder_input_name: input_tensor}
        output = self.encoder_session.run(self.encoder_output_names, inputs)
        self.encoder_out = np.array(output).flatten().astype(np.float32)

    def build_mode_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        now = time.monotonic()
        current_mode = self.manual_expert_mode
        if self.expert_cfg["auto_return_to_default"] and now >= self.jump_active_until:
            current_mode = self.expert_cfg["default_mode"]
        elif now < self.jump_active_until:
            current_mode = "jump"

        if current_mode == "jump":
            jump_active = 1.0 if now < self.jump_active_until else 0.0
            jump_target = self.jump_cfg["standing_height"] + self.jump_cfg["delta_height"] if jump_active > 0.5 else 0.0
            jump_commands = np.array(
                [jump_active, jump_target, self.jump_cfg["standing_height"]],
                dtype=np.float32,
            )
            gait_commands = np.zeros(self.gait_commands_size, dtype=np.float32)
            env_group = np.array([self.expert_cfg["jump_env_group"]], dtype=np.float32)
        else:
            jump_commands = np.zeros(self.jump_commands_size, dtype=np.float32)
            gait_commands = np.array(
                [
                    self.gait_cfg["frequency"],
                    self.gait_cfg["offset"],
                    self.gait_cfg["duration"],
                    self.gait_cfg["swing_height"],
                    self.gait_cfg["standing_height"],
                ],
                dtype=np.float32,
            )
            env_group = np.array([self.expert_cfg["gait_env_group"]], dtype=np.float32)

        return jump_commands, gait_commands, env_group

    def compute_actions(self) -> None:
        jump_commands, gait_commands, env_group = self.build_mode_inputs()

        if self.use_single_onnx:
            input_tensor = np.concatenate(
                [
                    self.proprio_history_buffer,
                    self.observations,
                    self.scaled_commands,
                    jump_commands,
                    gait_commands,
                    env_group,
                ],
                axis=0,
            ).astype(np.float32)
            inputs = {self.student_input_name: input_tensor}
            output = self.student_session.run(self.student_output_names, inputs)
        else:
            input_tensor = np.concatenate(
                [
                    self.encoder_out,
                    self.observations,
                    self.scaled_commands,
                    jump_commands,
                    gait_commands,
                    env_group,
                ],
                axis=0,
            ).astype(np.float32)
            inputs = {self.policy_input_name: input_tensor}
            output = self.policy_session.run(self.policy_output_names, inputs)

        self.actions = np.array(output).flatten().astype(np.float32)

    def set_joint_command(self, joint_index: int, q: float, dq: float, tau: float, kp: float, kd: float) -> None:
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self) -> None:
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()

        self.loop_count += 1
        self.robot.publishRobotCmd(self.robot_cmd)

    def robot_state_callback(self, robot_state: datatypes.RobotState) -> None:
        self.robot_state = robot_state

    def imu_data_callback(self, imu_data: datatypes.ImuData) -> None:
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    @staticmethod
    def _button_value(buttons, index: int | None) -> int:
        if index is None or index < 0 or index >= len(buttons):
            return 0
        return int(buttons[index])

    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy) -> None:
        buttons = list(sensor_joy.buttons)
        if not self.start_controller and self.calibration_state == 0 and buttons[4] == 1 and buttons[3] == 1:
            print("L1 + Y: start_controller...")
            self.start_controller = True
        if self.start_controller and buttons[4] == 1 and buttons[2] == 1:
            print("L1 + X: stop_controller...")
            self.start_controller = False

        if len(self.prev_buttons) != len(buttons):
            self.prev_buttons = [0 for _ in buttons]

        jump_pressed = self._button_value(buttons, self.jump_cfg["trigger_button_index"])
        prev_jump_pressed = self._button_value(self.prev_buttons, self.jump_cfg["trigger_button_index"])
        if jump_pressed == 1 and prev_jump_pressed == 0:
            self.jump_active_until = time.monotonic() + self.jump_cfg["hold_seconds"]
            if not self.expert_cfg["auto_return_to_default"]:
                self.manual_expert_mode = "jump"

        toggle_index = self.jump_cfg["expert_toggle_button_index"]
        toggle_pressed = self._button_value(buttons, toggle_index)
        prev_toggle_pressed = self._button_value(self.prev_buttons, toggle_index)
        if toggle_pressed == 1 and prev_toggle_pressed == 0:
            self.manual_expert_mode = "jump" if self.manual_expert_mode == "gait" else "gait"
            print(f"Expert mode toggled to: {self.manual_expert_mode}")

        linear_x = float(np.clip(sensor_joy.axes[1], -1.0, 1.0))
        linear_y = float(np.clip(sensor_joy.axes[0], -1.0, 1.0))
        angular_z = float(np.clip(sensor_joy.axes[2], -1.0, 1.0))

        self.commands[0] = linear_x * 0.5
        self.commands[1] = linear_y * 0.5
        self.commands[2] = angular_z * 0.5

        self.prev_buttons = buttons

    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue) -> None:
        if diagnostic_value.name == "calibration":
            print(f"Calibration state: {diagnostic_value.code}")
            self.calibration_state = diagnostic_value.code


def build_robot(robot_ip: str):
    robot = Robot(RobotType.PointFoot)
    if not robot.init(robot_ip):
        raise RuntimeError(f"Failed to initialize robot at {robot_ip}.")
    return robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wheelfoot multi-expert deploy controller.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to exported model directory.")
    parser.add_argument("--config", type=str, default=None, help="Optional params.yaml path.")
    parser.add_argument("--robot_ip", type=str, default="127.0.0.1", help="Robot IP address.")
    parser.add_argument("--robot_type", type=str, default="WF_TRON1A_MULTIEXPERT", help="Robot type label.")
    args = parser.parse_args()

    robot = build_robot(args.robot_ip)
    start_controller = args.robot_ip == "127.0.0.1"
    controller = WheelfootMultiExpertController(
        args.model_dir,
        robot,
        args.robot_type,
        start_controller=start_controller,
        config_path=args.config,
    )
    controller.run()


if __name__ == "__main__":
    main()


WFMultiExpertController = WheelfootMultiExpertController
