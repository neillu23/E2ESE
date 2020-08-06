import json

scp_file = "/mnt/Data/user_vol_2/user_neillu/TIMIT_fbank/timti_fbank.scp"

data_in = "/mnt/Data/user_vol_2/user_neillu/End2End/espnet/egs/timit/asr1/dump/test/deltafalse/data.json"
data_out = "test_data.json"

feature_map={}
with open(scp_file) as f:
    lines = f.readlines()
    for line in lines:
        k,v = line.split()
        feature_map[k] = v

with open(data_in) as json_file:
    data = json.load(json_file)
    for name in data["utts"]:
        for input_feats in data["utts"][name]["input"]:
            input_feats["feat"] = feature_map[name]


with open(data_out, 'w') as outfile:
    json.dump(data, outfile, indent=4)
