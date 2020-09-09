import os
import sys

directory = sys.argv[1]
out_list = sys.argv[2]
with open(out_list, 'w') as f:
    for root, directories, files in os.walk(directory):
        for filename in files:
            if ".WAV" in filename:
                filepath = os.path.join(root, filename)
                f.write(filepath+"\n")
