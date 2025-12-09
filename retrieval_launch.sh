#!/bin/bash

# Launch the dynamic BM25 retrieval server
# Documents are now provided dynamically with each query, so no index or corpus is needed

python search_r1/search/retrieval_server.py --host 127.0.0.1 \
                                            --port 56321 \
                                            --topk 3
