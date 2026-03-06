import mujoco
import mujoco.viewer

from zmq_common.utils import get_pub_socket, get_sub_socket
from zmq_common.publisher import ZmqPublisher
from zmq_common.subscriber import ZmqSubscriber

from dataclasses import dataclass

MS_TO_S = 1e-3
HOST = "localhost"
# Has to be inverse of the simulation driver
PUB_PORT = 5556
SUB_PORT = 5555
ROBOT_CMD_TOPIC = "robot_cmd"
ROBOT_STATUS_TOPIC = "robot_status"


@dataclass
class Robot:
    def __init__(self, models_dir):
        scene_file_path = f"{models_dir}/universal_robots_ur5e/scene.xml"
        gripper_file_path = f"{models_dir}/robotiq_2f85/2f85.xml"

        # Load scene (which already includes ur5e.xml internally)
        scene_spec = mujoco.MjSpec.from_file(scene_file_path)
        gripper_spec = mujoco.MjSpec.from_file(gripper_file_path)

        # Find attachment site and attach gripper
        attachment_site = next(s for s in scene_spec.sites if s.name == 'attachment_site')
        scene_spec.attach(gripper_spec, site=attachment_site, prefix='gripper_', suffix='')
        
        self.model = scene_spec.compile()
        self.data = mujoco.MjData(self.model)

    def get_joint_positions(self):
        return list(self.data.qpos[:7])

    def get_joint_velocities(self):
        return list(self.data.qvel[:7])

    def get_joint_accelerations(self):
        return list(self.data.qacc[:7])

def main():
    pub_socket = get_pub_socket(HOST, PUB_PORT)
    sub_socket = get_sub_socket(HOST, SUB_PORT)
    robot_cmd_sub = ZmqSubscriber(sub_socket, ROBOT_CMD_TOPIC)
    robot_status_pub = ZmqPublisher(pub_socket, ROBOT_STATUS_TOPIC)

    robot = Robot(models_dir="/home/kartik/dev/logbook_projects/mujoco_dev/mujoco_menagerie")

    # 1. Initialize target at the starting pose
    status_publish_interval_s = 2 * MS_TO_S

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        last_pub_time = robot.data.time
        while viewer.is_running():
            # Get command
            cmd = robot_cmd_sub.recv_message()
            if cmd is not None:
                cmd = cmd["value"]
                robot.data.ctrl[:7] = cmd["target_joint_positions"]
                    
            mujoco.mj_step(robot.model, robot.data)

            now = robot.data.time
            if now - last_pub_time >= status_publish_interval_s:
                robot_status = {
                    "time_s": robot.data.time,
                    "current_joint_positions": robot.get_joint_positions(),
                    "current_joint_velocities": robot.get_joint_velocities(),
                    "current_joint_accelerations": robot.get_joint_accelerations()
                }
                robot_status_pub.send_message(robot_status)
                last_pub_time = now

            viewer.sync()

if __name__ == "__main__":
    main()