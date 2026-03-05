import mujoco
import mujoco.viewer

from zmq_common.utils import get_pub_socket, get_sub_socket
from zmq_common.publisher import ZmqPublisher
from zmq_common.subscriber import ZmqSubscriber

MS_TO_S = 1e-3
HOST = "localhost"
# Has to be inverse of the simulation driver
PUB_PORT = 5556
SUB_PORT = 5555
ROBOT_CMD_TOPIC = "robot_cmd"
ROBOT_STATUS_TOPIC = "robot_status"

def main():
    pub_socket = get_pub_socket(HOST, PUB_PORT)
    sub_socket = get_sub_socket(HOST, SUB_PORT)
    robot_cmd_sub = ZmqSubscriber(sub_socket, ROBOT_CMD_TOPIC)
    robot_status_pub = ZmqPublisher(pub_socket, ROBOT_STATUS_TOPIC)

    # Mujoco specific stuff
    UR5_SCENE_PATH = "/home/kartik/dev/logbook_projects/mujoco_dev/mujoco_menagerie/universal_robots_ur5e/scene.xml"
    model =  mujoco.MjModel.from_xml_path(UR5_SCENE_PATH)
    data = mujoco.MjData(model)

    # 1. Initialize target at the starting pose
    status_publish_interval_s = 2 * MS_TO_S

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_pub_time = data.time
        while viewer.is_running():
            # Get command
            cmd = robot_cmd_sub.recv_message()
            if cmd is not None:
                cmd = cmd["value"]
                data.ctrl[:6] = cmd["target_joint_positions"]
                    
            mujoco.mj_step(model, data)

            now = data.time
            if now - last_pub_time >= status_publish_interval_s:
                robot_status = {
                    "time_s": data.time,
                    "current_joint_positions": list(data.qpos),
                    "current_joint_velocities": list(data.qvel),
                    "current_joint_accelerations": list(data.qacc)
                }
                robot_status_pub.send_message(robot_status)
                last_pub_time = now

            viewer.sync()

if __name__ == "__main__":
    main()