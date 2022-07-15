#! /home/riyuu/anaconda3/envs/daily/bin/python
import json
import rospy
from std_msgs.msg import String


def yolo_cb(msg: String):
    s = msg.data
    result = json.loads(s)
    print(result)
    # rospy.loginfo(f"消息yolo处理结果的订阅者，我接收到了:")
    # rospy.loginfo(f"{type(result)}, 他的值是: \n{result}")


if __name__ == "__main__":
    rospy.init_node("yolo_result_user")
    sub = rospy.Subscriber(
        name="yolo_result", data_class=String, queue_size=100, callback=yolo_cb)
    rospy.spin()
