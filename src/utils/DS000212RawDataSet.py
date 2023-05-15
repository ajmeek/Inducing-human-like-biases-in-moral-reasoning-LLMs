from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
from re import search
from csv import DictReader


class DS000212RawDataset(Dataset):
	"""
	Map-style dataset that loads ds000212 dataset with its scenarios from disk and
	prepares it for fine tuning.
	"""
	event_to_scenario = {
		"A_PHA": ("accidental", "Physical harm"),
		"B_PSA": ("accidental", "Psychological harm"),
		"C_IA": ("accidental", "Incest"),
		"D_PA": ("accidental", "Pathogen"),
		"E_NA": ("accidental", "Neutral"),
		"F_PHI": ("intentional", "Physical harm"),
		"G_PSI": ("intentional", "Psychological harm"),
		"H_II": ("intentional", "Incest"),
		"I_PI": ("intentional", "Pathogen"),
		"J_NI": ("intentional", "Neutral"),
	}

	def __init__(self, dataset_path : Path, scenarios_csv : Path, tokenizer):
		assert dataset_path.exists()
		assert scenarios_csv.exists()
		assert tokenizer
		self.target_head_dim = None
		self._dataset_path = dataset_path
		self._init_scenarios(scenarios_csv)
		assert any(self._scenarios)
		self._build_index()
		self._tokenizer = tokenizer
		assert any(self._inx_to_item)

	def __getitem__(self, index):
		npz_file, inner_inx = self._inx_to_item[index]
		assert npz_file
		assert npz_file.exists()
		loaded = np.load(npz_file)
		data_items = loaded['data_items']
		labels = loaded['labels']
		assert data_items.shape[0] == labels.shape[0] 
		assert labels.shape[0] > inner_inx
		label = labels[inner_inx]
		text = self._parse_label(label)
		assert text
		tokenized = self._tokenizer(text, padding='max_length', truncation=True)
		tokens = torch.tensor(tokenized['input_ids'])
		mask = torch.tensor(tokenized['attention_mask'])
		target = data_items[inner_inx]
		return tokens, mask, target

	def __len__(self):
		return len(self._inx_to_item)

	def _init_scenarios(self, scenarios_csv : Path):
		self._scenarios = []
		with open(scenarios_csv, newline='', encoding='utf-8') as csvfile:
			reader = DictReader(csvfile)
			for row in reader:
				self._scenarios.append(row)

	def _build_index(self):
		self._inx_to_item = []
		for npz_file in Path(self._dataset_path).glob('**/*.npz'):
			if not npz_file.exists():
				continue
			description_file = Path(str(npz_file).replace('.npz', '-description.txt'))
			assert description_file.exists(), description_file
			items_num = None
			# Read all lines in description_file into description_text:
			m = search(
				'data shape: \((\d+), *(\d+)\)',
				description_file.read_text()
			)
			if not m:
				continue
			items_num = int(m.group(1))
			head_dim = int(m.group(2))
			if not self.target_head_dim:
				self.target_head_dim = head_dim
			else:
				assert self.target_head_dim == head_dim, \
					 f"Expected normalized but {self.target_head_dim} != {head_dim}, {npz_file}"

			self._inx_to_item += [(npz_file, i ) for i in range(items_num)]
	
	def _parse_label(self, label) -> str:
		condition, item, key = label
		skind, stype = DS000212RawDataset.event_to_scenario[condition]
		found = [s for s in self._scenarios if s['item'] == item]
		if not found:
			return None
		found = found[0]
		assert found['type'] == stype, f"Scenario with {item} item does not match the '{stype}' expected type. Scenario: {found}. Event: {event}."
		text = ' '.join([
			found['background'],
			found['action'],
			found['outcome'],
			found[skind]
		])
		return text

