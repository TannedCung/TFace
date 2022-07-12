def fix(infile, outfile):
    with open(infile, 'r') as i_f: in_lines = i_f.readlines()
    with open(outfile, 'w') as o_f:
        for line in in_lines:
            o_f.write(line.replace("Dataoutsider_save_feature", "Data/outsider_save").replace(".npy", ".jpg").replace("_feature", ""))

if __name__=="__main__":
    IN = "/workspace/annotations/quality_.txt"
    OUT = "/workspace/annotations/quality.txt"
    fix(IN, OUT)