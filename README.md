[//]: # (Image References)

[image1]: ./images/pcl-pointcloud.png
[image2]: ./images/pcl-cluster.png
[image3]: ./images/test_1_confusionmatrix.png
[image4]: ./images/test_1_rviz.png
[image5]: ./images/test_2_confusionmatrix.png
[image6]: ./images/test_2_rviz.png
[image7]: ./images/test_3_confusionmatrix.png
[image8]: ./images/test_3_rviz.png

---

# PR2 Robotic Object Recognition & Picking Project

<p align="center"> <img src="./images/Screenshot 2018-11-10 20:46:14.png"> </p>

Utilized Passthrough and RANSAC filtering on the point-cloud data
![alt text][image1]

Utilized Euclidean Clustering to distinguish identified objects for pick & place
![alt text][image2]

```
roslaunch sensor_stick training.launch
rosrun sensor_stick capture_features.py
```
Output is training_set.sav file

```
rosrun sensor_stick train_svm.py
```
Output is Confusion Matrices and model.sav file

**Test World 1**

Confusion Matrix - Not Confused
![alt text][image3]

3 of 3 Objects Identified
![alt text][image4]

Output_1 YAML file included

# Links

* PCL documentation : http://strawlab.github.io/python-pcl/
* RANSAC algorithm : http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/FISHER/RANSAC/
* Outlier Removal (paper) : http://people.csail.mit.edu/changil/assets/point-cloud-noise-removal-3dv-2016-wolff-et-al.pdf
* Clustering Algorithm : http://bit.ly/clustering-tutorial
* Segmentation with NN (intro) : http://bit.ly/segmentation-intro-nn
