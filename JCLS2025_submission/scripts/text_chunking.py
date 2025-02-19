# -*- coding: utf-8 -*-

import os
#import numpy as np

corpus_path = r"JCLS2025_submission\gutenberg_subset\60_novels"

filenames = [os.path.join(corpus_path, fn) for fn in sorted(os.listdir(corpus_path))]

def split_text(filename, n_words):
    inputdata = open(filename, 'r', encoding='utf-8').read().split('\n')
    chunks = []
    current_chunk_words = []
    current_chunk_word_count = 0
    for line in inputdata:
        line_split = line.split('\t')
        if len(line_split) > 2:
            current_chunk_words.append(line_split[0])
            current_chunk_word_count += 1
            if current_chunk_word_count == n_words:
                chunks.append(' '.join(current_chunk_words))
                current_chunk_words = []
                current_chunk_word_count = 0
    chunks.append(' '.join(current_chunk_words) )
    return chunks

chunk_length = 500

chunks = []

for filename in filenames:
    chunk_counter = 0
    texts = split_text(filename, chunk_length)
    for text in texts:
        chunk = {'text': text, 'number': chunk_counter, 'filename': filename}
        chunks.append(chunk)
        chunk_counter += 1
       
output_dir = r'JCLS2025_submission\doc_60_' + str(chunk_length)

for chunk in chunks:
    basename = os.path.basename(chunk['filename'])
    fn = os.path.join(output_dir,
                      "{}{:04d}".format(basename.replace(".csv", '_'), chunk['number'])+'.txt')
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(chunk['text'])
