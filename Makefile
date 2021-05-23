format:
	@echo "Formatting Python code..."
	@autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive .
	@isort .
	@yapf -ir .
	@autopep8 --in-place --aggressive --aggressive --recursive .
	@echo "Formatting Rust code..."
	@cd src/engine && cargo fmt

build-benchmarks:
	@c++ benchmarks/normal.cpp -o benchmarks/normal.run
	@c++ benchmarks/lognormal.cpp -o benchmarks/lognormal.run

generate-normal:
	./benchmarks/normal.run $(SIZE)

generate-lognormal:
	./benchmarks/lognormal.run $(SIZE)