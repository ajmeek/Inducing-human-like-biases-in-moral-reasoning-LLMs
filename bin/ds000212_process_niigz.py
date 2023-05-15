DESCRIPTION='''
This script converts .nii.gz and .tsv files into .npy files
suitable for fine tuning a language model.
'''
from sys import argv
from pathlib import Path
import numpy as np
from csv import DictReader
from nilearn.masking import compute_epi_mask, apply_mask
import nilearn
import nilearn.maskers
import nibabel as nib
import argparse
from re import search

def main():
	args = get_args().parse_args()
	if not args.from_niigz.exists():
		print(f"File {args.from_niigz} not found.")
		exit(1)
	if not args.from_tsv.exists():
		print(f"File {args.from_tsv} not found.")
		exit(1)
	process(
		args.from_niigz,
		args.from_tsv, 
		args.to_npz,
		args.to_npz_description,
		is_roi=args.roi)
	

def process(
		from_niigz: Path,
		from_tsv: Path,
		to_npz: Path,
		to_npz_description: Path,
		is_roi: bool = False
	):
	fmri_data = extract_fmri_data(from_niigz, is_roi)
	# Read the from_tsv tsv file with columns: onset, duration, condition, item, key, RT:
	scenarios = []
	with open(from_tsv, newline='') as csvfile:
		scenarios = list(DictReader(csvfile, delimiter='\t', quotechar='"'))
	TR=2
	data_items=[]
	labels=[]
	for s in scenarios:
		onset = s['onset']
		try:
			if '[' in onset and ']' in onset:
				m = search('\d+', onset)
				assert m
				onset = int(m.group(0))
			else:
				onset = int(onset)
		except Exception as e:
			print(f'Failed to parse "onset" for {from_tsv}: {e}')
			continue
		duration = int(s['duration'])
		onset //= TR
		duration //= TR
		hemodynamic_lag = 8 // TR
		data = np.average(fmri_data[onset+hemodynamic_lag:onset+duration+hemodynamic_lag], axis=0)
		label = [s[k] for k in ('condition', 'item', 'key')]
		data_items.append(data)
		labels.append(label)
	data_items = np.array(data_items)
	labels = np.array(labels)

	# Save data_items and labels into .npz compressed numpy file:
	np.savez(
		to_npz,
		data_items=data_items,
		labels=labels,
	)
	with open(to_npz_description, 'w') as f:
		f.write(f'data shape: {data_items.shape}\n')
		f.write(f'labels shape: {labels.shape}\n')

_difumo_masker = None
def _get_difumo_masker():
	global _difumo_masker
	if not _difumo_masker:
		#for generating ROI analysis
		difumo_atlas = nilearn.datasets.fetch_atlas_difumo(dimension=64)
		_difumo_masker = nilearn.maskers.NiftiMapsMasker(difumo_atlas['maps'], resampling_target='data', detrend=True).fit()
	return _difumo_masker


def extract_fmri_data(bold_f, is_roi) -> np.ndarray:
	if is_roi:
		data = nib.load(bold_f)
		return _get_difumo_masker().transform(data)
	else:
		bold_symlink_f = str(bold_f.resolve())
		mask_img = compute_epi_mask(bold_symlink_f)
		return apply_mask(bold_symlink_f, mask_img)


def get_args() -> argparse.ArgumentParser:
	"""Get command line arguments"""
	parser = argparse.ArgumentParser(description=DESCRIPTION)
	parser.add_argument(
		'from_niigz',
		type=Path,
		help='File .nii.gz with fMRI data.'
	)
	parser.add_argument(
		'from_tsv',
		type=Path,
		help='File .tsv with description of fMRI data.'
	)
	parser.add_argument(
		'to_npz',
		type=Path,
		help='File .npz, the target where to put parsed data.'
	)
	parser.add_argument(
		'to_npz_description',
		type=Path,
		help='File with description of .npz file'
	)
	parser.add_argument(
		'--roi',
		help='If to make ROI.',
		action=argparse.BooleanOptionalAction
	)
	return parser


if __name__ == '__main__':
    main()

