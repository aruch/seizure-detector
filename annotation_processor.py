import re
import json
import os
import datetime

# Turn a hh:mm:ss string to number of seconds
# hh: piece is optional

HHMMSS_PAT = re.compile("(\d+)??:??(\d+):(\d+)$")


def hhmmss_to_seconds(s):
    m = HHMMSS_PAT.match(s)
    g = m.groups()
    s = 0
    n = len(g)
    for i in range(n):
        if g[n-i-1] is not None:
            s += 60**i * int(g[n-i-1])
    return s


# annotation file should be:
# {"video_id":
#    {"animal_id": ["hh:mm:ss", ...], "animal_id": ["hh:mm:ss", ...]}
# }
def load_seizure_annotations(file_path):
    with open(file_path) as f:
        annotations = json.load(f)
    for date, positions in annotations.items():
        for position, videos in positions.items():
            for vid_id, animal_seizures in videos.items():
                for animal_id, seizure_times in animal_seizures.items():
                    times_in_seconds = [hhmmss_to_seconds(x) for x in seizure_times]
                    annotations[date][position][vid_id][animal_id] = times_in_seconds  
    return annotations


def feature_filename_to_fields(filename):
    day, position, mouse_id, video_id, i, layer = filename[:-4].split("_")
    date = datetime.datetime.strptime(day, "%b %d %Y")
    std_date = date.strftime("%Y-%m-%d")

    return {
        "date": day,
        "std_date": std_date,
        "position": position,
        "animal_id": mouse_id,
        "video_file": video_id,
        "video_id": video_id[:-4],
        "start_time": max((int(i) * 600 - 10), 0),
        "video_chunk_id": int(i),
        "layer": layer
    }


def seizure_times_from_npz_filename(file_name, seizure_annotations):
    fields = feature_filename_to_fields(file_name)
    start_time = fields["start_time"]
    max_time = start_time + 600
    std_date = fields["std_date"]
    vid_id = fields["video_id"]
    if start_time != 0:
        max_time += 10
    video_seizure_times = seizure_annotations[std_date][fields["position"]][vid_id][fields["animal_id"]]
    return [t for t in video_seizure_times if t > start_time and t < max_time]


def fields_to_directory(base, fields):
    d = os.path.join(fields["date"], fields["position"])
    return os.path.join(base, d)
