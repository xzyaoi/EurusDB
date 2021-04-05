format:
	autoflake --in-place --remove-unused-variables --recursive .
	isort .
	yapf -ir .

docs:
	python3 -m pdoc --html --output-dir docs src/indexing
	mv docs/indexing/* docs/
	rm -rf docs/indexing
