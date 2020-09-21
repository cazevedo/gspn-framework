#! /usr/bin/env python3
# Standard libs
import os
from flask import Flask, render_template, jsonify, request, flash
from werkzeug.utils import secure_filename, redirect
import json
from threading import Thread, Event
# ROS libs
import rospy
import actionlib
import gspn_framework_package.msg
from gspn_framework_package.msg import GSPNFiringData
from gspn_framework_package.srv import CurrentPlace,CurrentPlaceResponse
# Files from my package
from gspn_framework_package import policy
from gspn_framework_package import gspn as pn
from gspn_framework_package import gspn_tools

app = Flask(__name__)  # create an app instance

ALLOWED_EXTENSIONS = set(['xml'])

GSPN_UPDATES = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    global my_pn
    with open('/home/pedro/catkin_ws/src/gspn_framework_package/ros/src/gspn_framework_package_ros/gspn_execution_input_simple.json') as f:
        json_data = json.load(f)

    tool = gspn_tools.GSPNtools()
    to_open = '/home/pedro/catkin_ws/src/gspn_framework_package/ros/src/gspn_framework_package_ros/' + json_data["gspn"]
    my_pn = tool.import_xml(to_open)[0]
    return render_template("gspn_visualization_home.html", data=my_pn)


@app.route("/return_gspn_updates")
def return_gspn_updates():
    global GSPN_UPDATES
    if len(GSPN_UPDATES) > 0:
        GSPN_UPDATES_TO_SEND = GSPN_UPDATES
        GSPN_UPDATES = []
        return jsonify(GSPN_UPDATES_TO_SEND)
    else:
        return jsonify(GSPN_UPDATES)


def online_visualization_update(msg):
    global GSPN_UPDATES
    GSPN_UPDATES.append(msg.transition)
    GSPN_UPDATES.append(msg.marking)


Thread(target=lambda: rospy.init_node('gspn_online_visualization', disable_signals=True)).start()
rospy.Subscriber("/TRANSITIONS_FIRED",GSPNFiringData, online_visualization_update, queue_size=10)
if __name__ == "__main__":
    app.run(debug=True)
