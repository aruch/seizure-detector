import re
import json

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
    for vid_id, positions in annotations.items():
        for position, days in positions.items():
            for day, animal_seizures in days.items():
                for animal_id, seizure_times in animal_seizures.items():
                    times_in_seconds = [hhmmss_to_seconds(x) for x in seizure_times]
                    annotations[vid_id][position][day][animal_id] = times_in_seconds        
    return annotations

def feature_filename_to_fields(file_name):
    day, position, mouse_id, video_id, i = file_name[:-4].split("_")
    return {
        "date": day,
        "position": position,
        "animal_id": mouse_id,
        "video_id": video_id,
        "start_time": max((int(i) * 600 - 10), 0)
    }

def seizure_times_from_npz_filename(file_name, seizure_annotations):
    fields = feature_filename_to_fields(file_name)
    start_time = fields["start_time"]
    max_time = start_time + 600
    if start_time != 0:
        max_time += 10
    video_seizure_times = seizure_annotations[fields["video_id"]][fields["position"]][fields["date"]][fields["animal_id"]]
    return [t for t in video_seizure_times if t > start_time and t < max_time]