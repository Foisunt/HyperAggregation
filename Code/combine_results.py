from utils import collect_res_dat
from pathlib import Path
from tqdm import tqdm
import sys

# gets a dir name, collects results, copies it to res_comb dir
def combine_results(name):
    p = Path("../results/"+name)
    p2 = Path("../results_comb/"+name+".pkl")
    df = collect_res_dat(p)
    df.to_pickle(p2)

if __name__ == "__main__":
    combine_results(sys.argv[1])