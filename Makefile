init:
ifeq ($(TRAVIS), true)
		pip install -r reqs/travis-requirements.txt
		pip install pandas==${PANDAS_VERSION}
		pip install numpy==${NUMPY_VERSION}
		pip freeze --local
endif

test:
	py.test reliability/ -rfs --cov=reliability --block=False --cov-report term-missing

pre:
	pre-commit run --all-files