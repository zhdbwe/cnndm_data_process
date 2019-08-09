import struct
from tensorflow.core.example import example_pb2


def transfer_binary2text(set_name):
	article_list, abstract_list = [], []
	file_name = "data/finished_files/{}.bin".format(set_name)
	with open(file_name, 'rb') as reader:
		while True:
			len_bytes = reader.read(8)
			if not len_bytes:
				break
			str_len = struct.unpack('q', len_bytes)[0]
			example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
			e = example_pb2.Example.FromString(example_str)
			article_text = e.features.feature['article'].bytes_list.value[0]
			abstract_text = e.features.feature['abstract'].bytes_list.value[0]
			article_list.append(article_text.decode())
			abstract_list.append(abstract_text.decode())
	assert len(article_list) == len(abstract_list)
	art_file = "data/finished_files/{}_article.txt".format(set_name)
	abs_file = "data/finished_files/{}_abstract.txt".format(set_name)
	with open(art_file, 'w', encoding='utf-8') as art_f, open(abs_file, 'w', encoding='utf-8') as abs_f:
		art_f.write('\n'.join(article_list))
		abs_f.write('\n'.join(abstract_list))


if __name__ == '__main__':
	for set_name in ['train', 'val', 'test']:
		transfer_binary2text(set_name)
