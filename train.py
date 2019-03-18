import json
import seizure_detector

feature_dir = "/run/media/aruch/glacier/video_features/"
video_dir = "/home/aruch/sherlock/mwintermark/rat_videos/"
model_dir = "/home/aruch/seizure-detection/models/"
annotation_file = "/home/aruch/seizure-detection/seizure_annotations.json"

training_files = set(json.load(open("training_files.json")))
dev_files = set(json.load(open("dev_files.json")))
hp_space = json.load(open("inception_rnn_hp_space.json"))

detector = seizure_detector.SeizureDetector("inception_rnn", model_dir)
detector.initialize_training_setup(feature_dir=feature_dir,
                                   video_dir=video_dir,
                                   annotation_file=annotation_file,
                                   train_files=training_files,
                                   dev_files=dev_files,
                                   files_per_epoch=80,
                                   file_pos_p=0.6,
                                   param_grid=hp_space,
                                   notebook=False)
detector.initialize_training_model(params)
detector.train(epochs=60, dev_stats_every=5)
