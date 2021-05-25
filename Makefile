format:
	@echo "Formatting Python code..."
	@autoflake --in-place --remove-unused-variables --remove-all-unused-imports --recursive .
	@isort .
	@yapf -ir .
	@autopep8 --in-place --aggressive --aggressive --recursive .
	@echo "Formatting Rust code..."
	@cd src/engine && cargo fmt

htmldocs:
	@cd docs && yarn && yarn build
	@echo "Build finished, server started. Now redirecting to browser..."
	@echo "If your browser is not open automatically, please open http://127.0.0.1:8000"
	@python -m webbrowser "http://127.0.0.1:8000"
	@cd docs/src/.vuepress/dist && python3 -m http.server

build-benchmarks:
	@c++ benchmarks/normal.cpp -o benchmarks/normal.run
	@c++ benchmarks/lognormal.cpp -o benchmarks/lognormal.run

generate-normal:
	./benchmarks/normal.run $(SIZE)

generate-lognormal:
	./benchmarks/lognormal.run $(SIZE)
