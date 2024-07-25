
from typing import Iterator

import pandas
import numpy

DEF_NUM_VARIANTS = 10000

class VariantsChunk():
	def __init__(self, variants_info:pandas.DataFrame, alleles:pandas.DataFrame, gts:numpy.array):
		...
	
	@property
	def num_variants(self):
		...
		
	@property
	def samples(self):
	# or indi_names
		...

	@property		
	def ploidy(self):
		...

	@property		
	def gt_array(self):
		...


class Variants:
	def init__(self, variant_chunks:Iterator[VariantsChunk])__:
		...
		
	def get_variant_chunks(self, num_variants_per_chunk=DEF_NUM_VARIANTS, keep_only_gts=False):
		...
		
	@property
	def num_variants(self):
		...
		
	@property
	def samples(self):
		...
