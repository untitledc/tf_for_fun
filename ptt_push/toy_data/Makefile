translate:
	python gen_target.py -s source.txt -r rule.txt > target.txt

gen_vocab:
	python gen_vocab.py -d source.txt --max 10000 > source_vocab.txt
	python gen_vocab.py -d target.txt > target_vocab.txt

gen_source:
	python gen_random_source.py -v source_vocab.txt > source.txt
