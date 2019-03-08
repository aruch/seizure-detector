import tqdm
import os
import time
import datetime
import math
import pickle
import json
import re

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from nets import inception

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import moviepy
import moviepy.editor
import imageio

# Function take a subclip object defined by moviepy and creates a numpy array of # size of the subclip over the number of frames (numofframes)
# Returns the numpy array that should be saved
def make_np_array_from_subclip(subclip, num_frames):
    
    # get dimensions of subclip
    [n_y, n_x, n_c] = subclip.get_frame(0).shape

    # Create placeholder for numpy array
    subclip_np = np.zeros((num_frames, n_y, n_x, n_c), dtype=np.uint8)

    # Iterate through slices of subclip and add to numpy array
    for nn, frame in enumerate(subclip.iter_frames()):
        subclip_np[nn, :,:,:] = frame
        
    return subclip_np

def write_features(array, day, position, mouse_id, video_id, start_time, i, 
                   layer, outdir):
    fname = "{:s}_{:s}_{:s}_{:s}_{:d}_{:s}.npz".format(day, position, mouse_id, video_id, i, layer)
    path = os.path.join(outdir, day, position, fname)
    np.savez(path, 
             features=array, 
             start_time=start_time)

def build_video_features(movie_dir, fname, date, position, 
                         animal_id, feature_folder,
                         cage_positions, sess, endpoints, seq_input,
                         write_size = 10 * 60,
                         overlap = 10):
    print("Building Video Feature Array for {:s}, {:s}, {:s}".format(fname, date, position))
    
    vid = moviepy.editor.VideoFileClip(os.path.join(movie_dir, fname))
    
    clip = moviepy.video.fx.all.crop(
        vid, x1=cage_positions["x1"], y1=cage_positions["y1"],
        x2=cage_positions["x2"], y2=cage_positions["y2"]
    )

    duration = vid.duration
    niters = math.ceil(duration/write_size)
    pbar = tqdm.tqdm(total=(vid.duration + (niters - 1) * overlap))

    for i in range(niters):
        clip_start = max(0, i * write_size - overlap)
        clip_end = min(clip.duration, (i + 1) * write_size)
        vid_clip = clip.subclip(clip_start, clip_end)
        last_layer = build_video_feature_array(sess, vid_clip, 
                                               end_points, seq_input,
                                               pbar=pbar)
        write_features(last_layer, date, position, animal_id, fname, clip_start,
                       i, "PreLogitsFlatten", feature_folder)

def build_video_feature_array(sess, clip, end_points, seq_input, window=10, pbar=None):
    duration = int(clip.duration)
    fps = round(clip.fps)
    frame_features_shallow = np.zeros((duration*fps, 1536), dtype=np.float32)

    for i in range(math.ceil(duration/window)):
        clip_start = i * window
        clip_end = min(duration, clip_start + window)
        frame_start = clip_start * fps
        frame_end = clip_end * fps
        vid_clip = clip.copy().subclip(clip_start, clip_end)
        np_feed = make_np_array_from_subclip(vid_clip, frame_end - frame_start)
        bottleneck_shallow = sess.run(end_points['PreLogitsFlatten'],
                                 feed_dict={seq_input: np_feed})
        frame_features_shallow[frame_start:frame_end, :] = bottleneck_shallow
        clip_start = clip_end
        
        if pbar is not None:
            pbar.update(window)
    
    return frame_features_shallow


image_size = inception.inception_v4.default_image_size

rat_video_folder = "/home/aruch/sherlock/mwintermark/rat_videos"
rat_video_feature_folder = "/home/aruch/seizure-detection/video_features"

with open("position_annotations.json") as f:
    position_annotations = json.load(f)

VID_PAT = re.compile("(\w+).(MPG|MP4)$")

dates = ["Jan " + str(x) + " 2019" for x in range(18, 32) if x != 19]


seq_input = tf.placeholder(tf.int8, (None, None, None, 3))
processed_images = tf.image.convert_image_dtype(seq_input, dtype=tf.float32)
processed_images = tf.image.resize_bilinear(processed_images, [image_size, image_size],
                                            align_corners=False)
processed_images = tf.subtract(processed_images, 0.5)
processed_images = tf.multiply(processed_images, 2.0)

# Create the model, use the default arg scope to configure the batch norm parameters.
with slim.arg_scope(inception.inception_v4_arg_scope()):
    logits, end_points = inception.inception_v4(processed_images, 
                                                num_classes=1001, 
                                                is_training=False)
    
    init_fn = slim.assign_from_checkpoint_fn(
        'inception_v4.ckpt',
        slim.get_model_variables('InceptionV4'))
    
sess =  tf.Session() 
init_fn(sess)
        
for date in os.listdir(rat_video_folder):
    if date not in dates:
        continue
    if not os.path.isdir(os.path.join(rat_video_feature_folder, date)):
        os.mkdir(os.path.join(rat_video_feature_folder, date))
    for position in os.listdir(os.path.join(rat_video_folder, date)):
        if position != "3":
            continue
        if position not in position_annotations:
            continue
        if not os.path.isdir(os.path.join(rat_video_feature_folder, date, position)):
            os.mkdir(os.path.join(rat_video_feature_folder, date, position))
        movie_dir = os.path.join(rat_video_folder, date, position)
        for fname in os.listdir(movie_dir):
            
            m = VID_PAT.match(fname)
            if not m:
                continue
            
            for animal_id, cage_positions in position_annotations[position].items():
                build_video_features(movie_dir, fname, date, position, 
                                     animal_id, rat_video_feature_folder,
                                     cage_positions, sess, end_points, seq_input)
