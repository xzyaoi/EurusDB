format:
	autoflake --in-place --remove-unused-variables --recursive .
	isort .
	yapf -ir .
	cd src/engine && cargo fmt