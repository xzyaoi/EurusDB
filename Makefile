format:
	@autoflake --in-place --remove-unused-variables --recursive .
	@isort .
	@yapf -ir .
	@cd src/engine && cargo fmt

build-benchmarks:
	@c++ benchmarks/normal.cpp -o benchmarks/normal.run
	@c++ benchmarks/lognormal.cpp -o benchmarks/lognormal.run