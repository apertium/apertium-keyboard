# Copyright 2019 Christo Kirov. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

def _read_words(filename):
	print('Reading %s...' % (filename), file=sys.stderr)
	with tf.io.gfile.GFile(filename, "r") as f:
		return f.read().strip().replace("\n", "<eos>").split()


def _build_vocab(filename):
	data = _read_words(filename)
	data += _read_words(filename + ".out")

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	id_to_word = dict(zip(range(len(words)), words))

	# Save the vocab to a file, ordered according to the index.
	with open('labels.txt', 'w', encoding='utf-8') as outfile:
		for w in words:
			outfile.write(w + "\n")

	return word_to_id


def _file_to_word_ids(filename, word_to_id):
	print('Converting %s to IDs' % (filename), file=sys.stderr)
	data_in = _read_words(filename)
	data_out = _read_words(filename + ".out")
	ids_in = [word_to_id[word] for word in data_in if word in word_to_id]
	ids_out = [word_to_id[word] for word in data_out if word in word_to_id]
	print('	 ', len(ids_in),ids_in[-1],'|', len(ids_out), ids_out[-1], file=sys.stderr)
	assert(len(ids_in) == len(ids_out))
	return [(x,y) for x, y in zip(ids_in, ids_out)]


def lm_raw_data(data_path=None):
	"""Load LM raw data from data directory "data_path".

	Reads LM text files, converts strings to integer ids,
	and performs mini-batching of the inputs.

	Args:
		data_path: string path to the directory where simple-examples.tgz has
			been extracted.

	Returns:
		tuple (train_data, valid_data, test_data, vocabulary)
		where each of the data objects can be passed to PTBIterator.
	"""

	train_path = os.path.join(data_path, "lm.train.txt")
	valid_path = os.path.join(data_path, "lm.valid.txt")
	test_path = os.path.join(data_path, "lm.test.txt")

	word_to_id = _build_vocab(train_path)
	train_data = _file_to_word_ids(train_path, word_to_id)
	valid_data = _file_to_word_ids(valid_path, word_to_id)
	test_data = _file_to_word_ids(test_path, word_to_id)
	vocabulary = len(word_to_id)
	return train_data, valid_data, test_data, vocabulary


def lm_producer(raw_data, batch_size, num_steps, name=None):
	"""Iterate on the raw PTB data.

	This chunks up raw_data into batches of examples and returns Tensors that
	are drawn from these batches.

	Args:
		raw_data: one of the raw data outputs from ptb_raw_data.
		batch_size: int, the batch size.
		num_steps: int, the number of unrolls.
		name: the name of this operation (optional).

	Returns:
		A pair of Tensors, each shaped [batch_size, num_steps]. The second element
		of the tuple is the same data time-shifted to the right by one.

	Raises:
		tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
	"""
	with tf.name_scope(name, "LMProducer", [raw_data, batch_size, num_steps]):
		raw_data_in = [cp[0] for cp in raw_data]
		raw_data_out = [cp[1] for cp in raw_data]
		raw_data_in = tf.convert_to_tensor(raw_data_in, name="raw_data", dtype=tf.int32)
		raw_data_out = tf.convert_to_tensor(raw_data_out, name="raw_data", dtype=tf.int32)

		data_len = tf.size(raw_data_in)
		batch_len = data_len // batch_size
		data_in = tf.reshape(raw_data_in[0 : batch_size * batch_len],
											[batch_size, batch_len])
		data_out = tf.reshape(raw_data_out[0 : batch_size * batch_len],
											[batch_size, batch_len])

		epoch_size = (batch_len - 1) // num_steps
		assertion = tf.compat.v1.assert_positive(
				epoch_size,
				message="epoch_size == 0, decrease batch_size or num_steps")
		with tf.control_dependencies([assertion]):
			epoch_size = tf.identity(epoch_size, name="epoch_size")

		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
		x = tf.strided_slice(data_in, [0, i * num_steps],
												 [batch_size, (i + 1) * num_steps])
		x.set_shape([batch_size, num_steps])
		y = tf.strided_slice(data_out, [0, i * num_steps + 1],
												 [batch_size, (i + 1) * num_steps + 1])
		y.set_shape([batch_size, num_steps])
		return x, y
