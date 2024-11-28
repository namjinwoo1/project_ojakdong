#!/usr/bin/env python

from websocket_server import WebsocketServer
import rospy
from geometry_msgs.msg import Twist
import json

# ROS Node Initialization
rospy.init_node('ros_websocket_server')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# WebSocket Event Handlers
def on_new_client(client, server):
    rospy.loginfo(f"New client connected: {client['id']}")

def on_client_left(client, server):
    rospy.loginfo(f"Client disconnected: {client['id']}")

def on_message(client, server, message):
    try:
        # Parse JSON message
        data = json.loads(message)
        rospy.loginfo(f"Received message: {data}")

        # If movement command received
        if 'linear' in data and 'angular' in data:
            twist = Twist()
            twist.linear.x = data['linear']
            twist.angular.z = data['angular']
            pub.publish(twist)
            rospy.loginfo(f"Published Twist: linear={data['linear']}, angular={data['angular']}")
        
        # Send acknowledgment to client
        server.send_message(client, json.dumps({"status": "Command received"}))
    except Exception as e:
        rospy.logerr(f"Error: {e}")

# Initialize WebSocket Server
server = WebsocketServer(host='0.0.0.0', port=9090)
server.set_fn_new_client(on_new_client)
server.set_fn_client_left(on_client_left)
server.set_fn_message_received(on_message)

rospy.loginfo("WebSocket server started on port 9090")
server.run_forever()
