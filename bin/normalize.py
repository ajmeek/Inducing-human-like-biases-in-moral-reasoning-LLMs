''' 
Normalises .npy files. Makes all .npy files with numbers to be 
of the same shape, with 0 mean, and 1 std.
'''
from sys import argv
from pathlib import Path
import numpy as np


def main():
    if len(argv) != 4 or not str.isdigit(argv[1]):
        print('Usage: normilize.py length <.npy file> <report file>')
        return

    max_l = int(argv[1])
    f = Path(argv[2])
    report = Path(argv[3])
    if not f.exists():
        return
    pad_args = {'mode': 'constant', 'constant_values': 0}

    in_files = np.load(f)
    data = in_files['data_items'].copy()
    labels = in_files['labels'].copy()

    # Normalize:
    data = (data - data.mean(axis=1, keepdims=True)) / \
        data.std(axis=1, keepdims=True)
    diff = max_l - data.shape[-1]
    data = np.pad(data, ((0, 0), (0, diff)), **pad_args)

    # Save:
    np.savez(f, data_items=data, labels=labels)
    to_npz_description = Path(str(f).replace('.npz', '-description.txt'))
    to_npz_description.write_text(
        f"data shape: {data.shape}\nlabels shape: {labels.shape}\n")
    report.write_text(
        f"Result shape: {data.shape}. Diff: {str(diff) if diff > 0 else ''}")


if __name__ == '__main__':
    main()
