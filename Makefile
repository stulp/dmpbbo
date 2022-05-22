all: build build_debug install

build:
	mkdir -p build_dir
	cd build_dir && cmake .. -DCMAKE_BUILD_TYPE=Release
	make -C build_dir -j 5 

build_debug:
	mkdir -p build_dir_debug
	cd build_dir_debug && cmake .. -DCMAKE_BUILD_TYPE=Debug -DREALTIME_CHECKS=1
	make -C build_dir_debug -j 5 

install: build
	make -C build_dir install
	
install_debug: build_debug
	make -C build_dir_debug install
	
test:
	python3 -m pytest tests/

format:	
	find -name '*.*pp' -exec clang-format -i -style=file {} \;
	find -name '*.py' -exec black -l 100 {} \;
	find -name '*.py' -exec isort -l 100 {} \;
	find -name '*.py' -exec autopep8 --select=W291,W293 --aggressive --in-place {} \;

lint:
	flake8
	
commit-checks: format lint

