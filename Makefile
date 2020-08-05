init:
ifeq ($(TRAVIS), true)
		pip install -r tests/travis-requirements.txt
		pip install pandas==${PANDAS_VERSION}
		pip install numpy==${NUMPY_VERSION}
		pip freeze --local
endif

test:
	py.test reliability/ -rfs --cov=reliability --block=False --cov-report term-missing

lint:
ifeq ($(TRAVIS_PYTHON_VERSION), 2.7)
		echo "Skip linting for Python2.7"
else
		make black
		prospector --output-format grouped
endif

black:
ifeq ($(TRAVIS_PYTHON_VERSION), 2.7)
		echo "Skip linting for Python2.7"
else
		black reliability/ -l 120 --fast
endif

check_format:
ifeq ($(TRAVIS_PYTHON_VERSION), 3.6)
		black . --check --line-length 120
else
		echo "Only check format on Python3.6"
endif

pre:
	pre-commit run --all-files