import json
import sys

#$VC_ESPNET_PATH/dump/dysarthric_all_eval_cospro/data.json
data_vc_in1 = sys.argv[1] 
data_vc_in2 = sys.argv[2]
data_vc_in3 = sys.argv[3]
data_asr_in = sys.argv[4] #"/mnt/Data/user_vol_2/user_neillu/End2End/espnet/egs/timit/asr1/dump/test/deltafalse/data.json"

data_out = sys.argv[5] #"test_data.json"

def readjson(data_vc_in, feature_map, shape_map):
    print("reading...",data_vc_in)
    with open(data_vc_in) as json_file_vc:
        data_vc = json.load(json_file_vc)
        for name in data_vc["utts"]:
            for vc_output_feats in data_vc["utts"][name]["output"]:
                feature_map[name] = vc_output_feats["feat"]
                shape_map[name] = vc_output_feats["shape"]

    return feature_map, shape_map

feature_map={}
shape_map={}
feature_map, shape_map = readjson(data_vc_in1, feature_map, shape_map)
feature_map, shape_map = readjson(data_vc_in2, feature_map, shape_map)
feature_map, shape_map = readjson(data_vc_in3, feature_map, shape_map)

with open(data_asr_in) as json_file_asr:
    data_asr = json.load(json_file_asr)
    notfound=[]
    for name in data_asr["utts"]:
        for input_feats in data_asr["utts"][name]["input"]:
            num=int(name.split("_")[-1])
            key=name.split("_")[1]+"_%02d%02d"%((num-1)/10+1,(num-1)%10+1)
            if key not in feature_map:
                notfound.append(name)
                continue
            input_feats["feat"] = feature_map[key]
            input_feats["shape"] = shape_map[key]
    for name in notfound:
        del data_asr["utts"][name]
        print(name,"not in vc dictionary, deleted!")



with open(data_out, 'w') as outfile:
    json.dump(data_asr, outfile, indent=4)

print("Saved to ", data_out)
