#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, yaml_dict_list):
    data_dict = {"object_list": yaml_dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Function to search list of dictionaries and return a selected value in selected dictionary
def search_dictionaries(key1, value1, key2, list_of_dictionaries):
    selected_dic = [element for element in list_of_dictionaries if element[key1] == value1][0]
    selected_val = selected_dic.get(key2)
    return selected_val

# Callback function for your Point Cloud Subscriber
def pcl_callback(ros_pcl_msg):
    
    # Convert a ROS PointCloud2 message to a pcl PointXYZRGB
    cloud_filtered = ros_to_pcl(ros_pcl_msg)
    # Create a statistical filter object: 
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(3)
    # Set threshold scale factor
    x = 0.00001
    # Any point with a mean distance larger than global (mean distance+x*std_dev)
    # will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Call the filter function
    cloud_filtered = outlier_filter.filter()
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.005   
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()
    
    # PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6095
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    
    # PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.456
    axis_max =  0.456
    passthrough.set_filter_limits(axis_min, axis_max) 
    cloud_filtered = passthrough.filter()
    
    # RANSAC plane segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.006
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    extracted_table   = cloud_filtered.extract(inliers, negative=False)
    extracted_objects = cloud_filtered.extract(inliers, negative=True)
    
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(9000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # Converts a pcl PointXYZRGB to a ROS PointCloud2 message
    ros_cloud_objects = pcl_to_ros(extracted_objects)
    ros_cloud_table   = pcl_to_ros(extracted_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        pcl_cluster = extracted_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        c_hists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        n_hists = compute_normal_histograms(normals)
        feature = np.concatenate((c_hists, n_hists))
        
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .2
        object_markers_pub.publish(make_label(label,label_pos, index))

        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
        
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    detected_objects_pub.publish(detected_objects)
    
    if len(detected_objects)>0:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else:
        rospy.logwarn('No detected objects !!!')
    
    return
    

def pr2_mover(object_list):

    # Vars
    test_scene_num = Int32()
    object_name    = String()
    pick_pose      = Pose()
    place_pose     = Pose()
    arm_name       = String()
    yaml_dict_list = []
    
    # Update test scene number based on the selected test.
    test_scene_num.data = 3

    # Get Params
    object_list_param = rospy.get_param('/object_list')
    dropbox_param     = rospy.get_param('/dropbox')

    '''
    # Rotate PR2 in place to capture side tables for the collision map
    # Rotate Right
    pr2_base_mover_pub.publish(-1.57)
    rospy.sleep(15.0)
    # Rotate Left
    pr2_base_mover_pub.publish(1.57)
    rospy.sleep(30.0)
    # Rotate Center
    pr2_base_mover_pub.publish(0)
    '''

    # Object Centroids
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])

    # Pick List For Loop
    for i in range(0, len(object_list_param)):
        
        
        # Read object name and group from object list.
        object_name.data = object_list_param[i]['name' ]
        object_group     = object_list_param[i]['group']

        # Select pick pose
        try:
            index = labels.index(object_name.data)
        except ValueError:
            print "Object not detected: %s" %object_name.data
            continue

        pick_pose.position.x = np.asscalar(centroids[index][0])
        pick_pose.position.y = np.asscalar(centroids[index][1])
        pick_pose.position.z = np.asscalar(centroids[index][2])

        # Select place pose
        position = search_dictionaries('group', object_group, 'position', dropbox_param)
        place_pose.position.x = position[0]
        place_pose.position.y = position[1]
        place_pose.position.z = position[2]

        # Select the arm to be used for pick_place
        arm_name.data = search_dictionaries('group', object_group, 'name', dropbox_param)

        # Create a list of dictionaries for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)

        rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output YAML
    yaml_filename = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(yaml_filename, yaml_dict_list)

    return

if __name__ == '__main__':

    # ROS Node Init
    rospy.init_node('clustering', anonymous=True)

    # Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Publishers
    pcl_objects_pub      = rospy.Publisher("/pcl_objects"     , PointCloud2,          queue_size=1)
    pcl_table_pub        = rospy.Publisher("/pcl_table"       , PointCloud2,          queue_size=1)
    pcl_cluster_pub      = rospy.Publisher("/pcl_cluster"     , PointCloud2,          queue_size=1)
    object_markers_pub   = rospy.Publisher("/object_markers"  , Marker,               queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pr2_base_mover_pub   = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    get_color_list.color_list = []

    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    while not rospy.is_shutdown():
        rospy.spin()
