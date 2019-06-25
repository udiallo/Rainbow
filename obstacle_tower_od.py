import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import matplotlib.pyplot as plt


class ObjectDetection:

    def __init__(self):
        # if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
        #    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
        # List of the strings that is used to add correct label for each box.
        self.category_index = label_map_util.create_category_index_from_labelmap('labelmap.txt', use_display_name=True)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile("nn1.pb", 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


    def run_inference_for_images(self, img, graph):
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, img.shape[0], img.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(img, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                return output_dict


    def detect_objects(self, input_image, im_save=False, dir='png', im_id=0):
        # Actual detection
        output_dict = self.run_inference_for_images(input_image, self.detection_graph)
        # Visualization of the results of a detection.
        doors = []
        for box, classe, score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
            # if score > 0.5:
            if score > 0.33:
                print(box, self.category_index[classe]['name'], round(score))
                doors.append([box, self.category_index[classe]['name'], round(score)])
        if im_save:
            # input_image = np.zeros((168, 168, 3))
            image_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
                input_image,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=0.50)
            plt.imsave(dir + str(im_id-1).zfill(5) + '_od.jpg', image_boxes, cmap='plasma')
        return doors


    def detect_objects_list(self, image_np):
        # Actual detection
        output_dict = self.run_inference_for_images(image_np, self.detection_graph)
        # Visualization of the results of a detection.
        doors = []
        for box, classe, score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
            if score > 0.5:
                print(box, self.category_index[classe]['name'], round(score))
                doors.append([box, self.category_index[classe]['name'], round(score)])
        return doors
