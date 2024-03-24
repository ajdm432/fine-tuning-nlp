from datasets import load_dataset
import random
import argparse

def save_fraction_dataset(dataset_name, fraction, save_name, subset=None):
    if subset is not None:
        data = load_dataset(dataset_name, name=subset, cache_dir=None)
    else:
        data = load_dataset(dataset_name, cache_dir=None)

    splits = data.keys()
    for split in splits:
        splitdata = data[split]
        num_to_keep = int(len(splitdata) * fraction)
        indices = list(range(len(splitdata)))
        # randomly shuffle if you want
        # random.shuffle(indices)
        data[split] = splitdata.select(indices[:num_to_keep])

    # save data
    data.save_to_disk(save_name)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataname", dest="dataname", default="")
    argparser.add_argument("-f", "--datafrac", dest="fraction", default=0.1, type=float)
    argparser.add_argument("-n", "--savename", dest="savename", default="data/dataset")
    argparser.add_argument("-s", "--subset", dest="subset", default=None)
    opts = argparser.parse_args()
    save_fraction_dataset(dataset_name=opts.dataname, fraction=opts.fraction, save_name=opts.savename, subset=opts.subset)