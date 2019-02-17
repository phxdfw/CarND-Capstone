import rospy
import os
import glob
import numpy as np
import cv2
import tensorflow as tf

STATUS = ['Red', 'Yellow', 'Green', '', 'Unknown']
THRESHOLD = 0.2

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()

        inference_path='light_classification/frozen_inference_graph.pb'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(inference_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0') 

    def get_classification(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            boxes, scores, classes, num_detections = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded}
            )
            max_score = 0
            max_label = 0
            for score, class_label in zip(
                np.squeeze(scores), np.squeeze(classes)
            ):
                if score > max_score:
                    max_score = score
                    max_label = class_label

            class_label = int(max_label)
            class_label = 4 if class_label == 4 else class_label - 1

            # confidence too low for either class
            if max_score < THRESHOLD:
                class_label = 4
                max_score = 1 - 3 * max_score

            rospy.loginfo(
                "TLClassifier: {0} with score: {1}".format(
                    STATUS[class_label], max_score
                )
            )
            return class_label;

        return 4
