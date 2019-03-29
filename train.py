import json
import seizure_detector

feature_dir = "/run/media/aruch/glacier/video_features/"
video_dir = "/Users/aruch/sherlock/wintermark_scratch/rat_videos/"
model_dir = "/Users/aruch/seizure-detector/models/"
annotation_file = "/Users/aruch/seizure-detector/seizure_annotations.json"

training_files = set(json.load(open("training_files.json")))
dev_files = set(json.load(open("dev_files.json")))
hp_space = json.load(open("inception_rnn_hp_space.json"))

detector = seizure_detector.SeizureDetector("3d_conv", model_dir)
detector.initialize_training_setup(feature_dir=feature_dir,
                                   video_dir=video_dir,
                                   annotation_file=annotation_file,
                                   train_files=training_files,
                                   dev_files=dev_files,
                                   files_per_epoch=16,
                                   file_pos_p=0.6,
                                   param_grid=hp_space,
                                   notebook=False)
detector.initialize_training_model()
detector.train(epochs=60, dev_stats_every=5)
