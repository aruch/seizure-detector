"""
Neural network model for rat/moust seizure detection
"""
import logging
import annotation_processor
import numpy as np
import random
import os
import threading
import uuid
import tqdm
import math

import tensorflow as tf
import json


class SeizureDetector:
    def __init__(self, architecture="inception_rnn", model_dir="models"):
        self.architecture = architecture
        self.model_dir = model_dir
        self.loaders = {}
        self.epoch_example_dict = {}

    def load_feature_files(self, filenames, epoch):
        examples = {}
        for filename in filenames:
            file_fields = annotation_processor.feature_filename_to_fields(filename)
            full_path = annotation_processor.fields_to_directory(self.feature_dir, file_fields)
            try:
                with np.load(os.path.join(full_path, filename)) as data:
                    example = {}
                    example["features"] = data["features"]
                    example["start_time"] = data["start_time"]
                    examples[filename] = example
            except Exception as e:
                print(filename)
                print(e)

        self.epoch_example_dict[epoch] = examples

    def initialize_training_setup(self, feature_dir, video_dir,
                                  annotation_file, train_files, dev_files,
                                  files_per_epoch, file_pos_p=0.6,
                                  param_grid=None, notebook=True):
        self.notebook = notebook
        if notebook:
            self.tqdm_func = tqdm.tqdm_notebook
        else:
            self.tqdm_func = tqdm.tqdm
        self.feature_dir = feature_dir
        self.video_dir = video_dir
        self.seizure_annotations = annotation_processor.load_seizure_annotations(annotation_file)
        self.train_files = list(train_files)
        self.dev_files = list(dev_files)

        self.files_per_epoch = files_per_epoch
        self.file_pos_p = file_pos_p

        self.param_grid = param_grid

    def hp_sample(self, space):
        log_sample = "log_sample" in space and space["log_sample"]
        if log_sample:
            min_val = np.log(space["min"])
            max_val = np.log(space["max"])
        else:
            min_val = space["min"]
            max_val = space["max"]

        val = np.random.uniform(low=min_val, high=max_val)

        if log_sample:
            val = np.exp(val)

        return val

    def get_random_params(self):
        params = {}
        for param, values in self.param_grid.items():
            if isinstance(values, list):
                val = random.choice(values)
            else:
                val = self.hp_sample(values)
            params[param] = val
        return params

    def initialize_training_model(self, params=None, model_name=None):
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = str(uuid.uuid4())

        log_filename = os.path.join(self.model_dir, self.model_name + ".log")
        logging.basicConfig(
            filename=log_filename, filemode="w",
            level=logging.INFO
        )

        self.G = tf.Graph()

        if params is None:
            params = self.get_random_params()

        print("building model with:")
        print(json.dumps(params))

        self.pos_p = params["minibatch_proportion"]
        self.batch_size = params["minibatch_size"]
        self.window_length = params["window_size"]

        logging.info("params: {0:}".format(json.dumps(params)))

        del params["minibatch_proportion"]
        del params["minibatch_size"]
        del params["window_size"]

        with self.G.as_default():
            if self.architecture == "inception_rnn":
                self.build_inception_rnn_net(**params)
            elif self.architecture == "3d_conv":
                self.build_3d_conv_net(**params)

    def start_feature_load(self, epoch):
        if isinstance(epoch, str) and epoch == "dev":
            filenames = self.dev_files
        else:
            filenames = self.choose_epoch_files(self.files_per_epoch)
        self.loaders[epoch] = threading.Thread(target=self.load_feature_files,
                                               args=(filenames, epoch))
        self.loaders[epoch].start()

    def set_n_minibatches(self):
        n_positive_examples = len(self.pos_example_indices)
        x = n_positive_examples/self.pos_p
        if self.rnn_size == 512:
            z = 16
        elif self.rnn_size == 256:
            z = 4
        elif self.rnn_size == 128:
            z = 2
        else:
            z = 1
        z *= self.batch_size / 64
        z *= self.window_length / 10

        self.n_minibatches = min(int(x/self.batch_size) * 5, int(1000/z))

    def setup_dev(self):
        epoch = "dev"
        self.loaders[epoch].join()
        self.dev_examples = self.epoch_example_dict[epoch]
        self.epoch_positive_negative_times(dev=True)

    def setup_epoch(self, epoch):
        self.loaders[epoch].join()
        self.epoch_examples = self.epoch_example_dict[epoch]
        self.epoch_filenames = list(self.epoch_examples.keys())

        if epoch > 0:
            del self.epoch_example_dict[epoch - 1]

        self.epoch_positive_negative_times()
        self.set_n_minibatches()

    def get_dev_stats(self):
        self.setup_dev()
        dev_set_error = 0.0
        n_dev_examples = len(self.pos_example_indices) + len(self.neg_example_indices)
        n_dev_batches = math.ceil(n_dev_examples/self.batch_size)
        conf_matrix = np.zeros((2, 2))
        for batch_i in self.tqdm_func(range(n_dev_batches)):
            offset = batch_i * self.batch_size
            size = min(self.batch_size, n_dev_examples - offset)

            batch_X = np.zeros((size, self.window_length * 30, 1536))
            batch_Y = np.zeros((size, 1))

            for i in range(size):
                if i + offset < len(self.pos_example_indices):
                    pair = self.pos_example_indices[i + offset]
                    batch_Y[i] = 1
                else:
                    pair = self.neg_example_indices[
                        i + offset - len(self.pos_example_indices)
                    ]
                    batch_Y[i] = 0
                    batch_X[i, :, :] = self.indices_to_example(pair, dev=True)

            predictions, err = self.sess.run([self.pred, self.error], {
                self.inputs: batch_X,
                self.outputs: batch_Y
            })
            predictions = predictions.flatten()
            batch_Y = batch_Y.flatten()

            tp = np.sum((predictions >= 0.5) & (batch_Y == 1))
            fp = np.sum((predictions >= 0.5) & (batch_Y == 0))
            tn = np.sum((predictions < 0.5) & (batch_Y == 0))
            fn = np.sum((predictions < 0.5) & (batch_Y == 1))
            dev_set_error += err
            conf_matrix[0, 0] += tn
            conf_matrix[1, 0] += fn
            conf_matrix[0, 1] += fp
            conf_matrix[1, 1] += tp

        return dev_set_error, conf_matrix

    def run_epoch(self, update_every=50):
        epoch_error = 0.0
        epoch_loss = 0.0

        epoch_bar = self.tqdm_func(total=self.n_minibatches)
        epoch_bar.set_postfix(error=0.0, loss=0.0)

        for batch_i in range(self.n_minibatches):
            batch_X, batch_Y = self.generate_minibatch()
            feed_dict = {
                self.inputs: batch_X,
                self.outputs: batch_Y,
            }
            batch_error, batch_loss, _ = self.sess.run([self.error,
                                                        self.loss,
                                                        self.train_fn],
                                                       feed_dict)
            epoch_error += batch_error
            epoch_loss += batch_loss
            epoch_bar.update(1)
            if (
                    ((batch_i + 1) % update_every == 0) or
                    ((batch_i + 1) == self.n_minibatches)
            ):
                epoch_bar.set_postfix(error=epoch_error/batch_i,
                                      loss=epoch_loss/batch_i)

        epoch_bar.close()

        epoch_error /= self.n_minibatches
        epoch_loss /= self.n_minibatches

        return epoch_error, epoch_loss

    def train(self, epochs, dev_stats_every=5):
        self.start_feature_load(0)
        self.start_feature_load("dev")
        self.training_bar = self.tqdm_func(total=epochs)
        best_f1 = 0.0

        with tf.Session(graph=self.G) as sess:
            self.sess = sess

            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                if epoch + 1 < epochs:
                    self.start_feature_load(epoch + 1)

                self.setup_epoch(epoch)
                train_err, train_loss = self.run_epoch()
                if (epoch + 1) % dev_stats_every == 0:
                    self.setup_dev()
                    dev_error, conf_matrix = self.get_dev_stats()
                    true_pos = conf_matrix[1, 1] + 1e-6
                    precision = true_pos/(conf_matrix[0, 1] + true_pos)
                    recall = true_pos/(conf_matrix[1, 0] + true_pos)
                    dev_f1 = 2*((precision * recall) / (precision + recall))
                    if self.notebook:
                        print("Dev error: {0:.3f}, F1: {1:.3f}".format(dev_error, dev_f1))
                        print(conf_matrix)
                        
                    to_log = {
                        "epoch": epoch,
                        "dev_error": dev_error,
                        "dev_f1": dev_f1,
                        "dev_precision": precision,
                        "dev_confusion": str(conf_matrix),
                        "dev_recall": recall,
                        "train_error": train_err,
                        "train_loss": train_loss
                    }
                    logging.info(json.dumps(to_log))

                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        self.save_model(str(epoch))

                self.training_bar.update(1)

    def save_model(self, postfix):
        save_dir = os.path.join(self.model_dir, self.model_name)
        save_path = os.path.join(save_dir, postfix + ".ckpt")

        self.saver.save(self.sess, save_path)

    def load_seizure_annotations(self, filename):
        self.seizure_annotations = annotation_processor.load_seizure_annotations(filename)

    def file_annotated(self, filename):
        fields = annotation_processor.feature_filename_to_fields(filename)
        if fields["std_date"] not in self.seizure_annotations:
            return False
        nested_dict = self.seizure_annotations[fields["std_date"]]

        if fields["position"] not in nested_dict:
            return False
        nested_dict = nested_dict[fields["position"]]

        if fields["video_id"] not in nested_dict:
            return False
        nested_dict = nested_dict[fields["video_id"]]

        return fields["animal_id"] in nested_dict
    
    def choose_epoch_files(self, n_files):
        n_pos = int(n_files * self.file_pos_p)
        n_neg = n_files - n_pos

        positive_files = []
        negative_files = []
        for name in self.train_files:
            if not self.file_annotated(name):
                continue
            if len(annotation_processor.seizure_times_from_npz_filename(name, self.seizure_annotations)) > 0:
                positive_files.append(name)
            else:
                negative_files.append(name)

        positive_examples = list(np.random.choice(positive_files, size=n_pos))
        negative_examples = list(np.random.choice(negative_files, size=n_neg))

        return positive_examples + negative_examples
    
    # Function to generate vector y corresponding to  binary classification of
    # video clips intervals of duration = window_length annotation in seconds
    # seizure array will receive dictionary of video name of seizure times
    def ground_truth_label(self, seizure_array, window_start):
        # Check if sliding window overlaps with seizure window
        for k in seizure_array:
            # Here just hard-coded 10 sec as minimum duration of seizure
            if (window_start + self.window_length > k) and (window_start < k):
                return 1
            # Windows after the 10 sec minimum duration of seizure and less than 120 secs after seizure start
            if (window_start >= k) and (window_start < k + 120):
                return -1

        # Return 0 for non-seizure windows
        return 0
    
    def epoch_positive_negative_times(self, dev=False, fps=29.97):
        # wait until we've loaded all examples
        pos_example_indices = []
        neg_example_indices = []

        if dev:
            filenames = self.dev_files
            examples = self.dev_examples
        else:
            filenames = self.epoch_filenames
            examples = self.epoch_examples

        for file_idx, file_name in enumerate(filenames):
            processed_chunk = examples[file_name]
            # video times (sec)
            vid_start_time = processed_chunk["start_time"]
            vid_length = int(processed_chunk["features"].shape[0]/fps)
            seizure_times = annotation_processor.seizure_times_from_npz_filename(file_name, self.seizure_annotations)
            for i in range(vid_length - self.window_length):
                label = self.ground_truth_label(seizure_times,
                                                vid_start_time + i)
                if label == 0:
                    neg_example_indices.append((file_idx, i))
                elif label == 1:
                    pos_example_indices.append((file_idx, i))

        self.pos_example_indices = pos_example_indices
        self.neg_example_indices = neg_example_indices

    def indices_to_example(self, pair, dev=False, fps=30):
        file_index = pair[0]
        time_index = pair[1]
        if dev:
            example_index = self.dev_files[file_index]
            features = self.dev_examples[example_index]['features']
        else:
            example_index = self.epoch_filenames[file_index]
            features = self.epoch_examples[example_index]['features']
        return features[time_index:time_index+self.window_length*fps]

    def generate_minibatch(self):
        num_pos_examples = int(self.batch_size*self.pos_p)
        num_neg_examples = self.batch_size - num_pos_examples

        pos_examples = random.choices(self.pos_example_indices,
                                      k=num_pos_examples)
        neg_examples = random.choices(self.neg_example_indices,
                                      k=num_neg_examples)

        batch_x = np.zeros((self.batch_size, self.window_length * 30, 1536))
        batch_y = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            if i < num_pos_examples:
                pair = pos_examples[i]
                batch_y[i] = 1
            else:
                pair = neg_examples[i - num_pos_examples]
                batch_y[i] = 0
            batch_x[i, :, :] = self.indices_to_example(pair)

        return batch_x, batch_y

    def build_inception_rnn_net(self, rnn_size,
                                false_neg_pen,
                                learning_rate,
                                reg_pen,
                                input_size=1536,
                                output_size=1):
        """
        Minimally sets inputs, outputs, pred, error, loss, train_fn, and saver
        to be used elsewhere in the coded
        """
        self.rnn_size = rnn_size
        inputs = tf.placeholder(tf.float32, (None, None, input_size))
        outputs = tf.placeholder(tf.float32, (None, output_size))

        cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)

        initial_state = cell.zero_state(tf.shape(inputs)[0], tf.float32)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state
        )

        pred = tf.layers.dense(
            rnn_states[1], output_size, activation=tf.sigmoid
        )

        error = tf.add(
            tf.multiply(
                tf.multiply(outputs, -tf.log(pred)),
                false_neg_pen
            ),
            tf.multiply(
                tf.subtract(1.0, outputs),
                -tf.log(tf.subtract(1.0, pred))
            )
        )
        error = tf.reduce_mean(error)
        t_vars = tf.trainable_variables()
        non_bias = [tf.nn.l2_loss(v) for v in t_vars if 'bias' not in v.name]
        reg_loss_l2 = tf.add_n(non_bias) * reg_pen
        loss = reg_loss_l2 + error

        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_fn = opt.minimize(loss)

        self.inputs = inputs
        self.outputs = outputs
        self.pred = pred
        self.error = error
        self.loss = loss
        self.train_fn = train_fn
        self.saver = tf.train.Saver()
