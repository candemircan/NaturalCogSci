import os
from os.path import join
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

from NaturalCogSci.helpers import get_project_root
from NaturalCogSci.rsatools import cka


def main(args):

    project_root = get_project_root()
    if args.features == "all":
        feature_list = glob.glob(
            join(project_root, "data", "features", "*.txt")
        )
        feature_list.remove(join(project_root, "data", "features", "file_names.txt"))
    else:
        feature_list = [join(project_root, "data", "features", f"{args.features}.txt")]

    df_feature_list = []
    df_cka_list = [] 
    for feature in tqdm(feature_list):
        cka_value = cka(X=np.loadtxt(feature),
                    Y=np.loadtxt(
            join(project_root, "data", "features", f"{args.target}.txt")))
        df_feature_list.append(feature.split(os.sep)[-1].split(".txt")[0])
        df_cka_list.append(cka_value)




    df = pd.DataFrame({"feature": df_feature_list, "cka": df_cka_list})
    file_name = join(project_root, "data", "cka", f"target_{args.target}.csv")
    df.to_csv(
        file_name,
        index=False)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", "-f")
    parser.add_argument("--target", "-t")

    args = parser.parse_args()

    main(args)

