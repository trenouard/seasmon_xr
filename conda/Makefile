HDC_INDEX=../../hdc-conda-index

.PHONY: all
all: refresh-index

.PHONY: build
build:
	conda build --no-test --no-anaconda-upload --output-folder ./build recipe

.PHONY: refresh-index
refresh-index: build
	conda index -s noarch -n hdc $(HDC_INDEX)
	cp ./build/noarch/*tar.bz2 $(HDC_INDEX)/noarch/
	conda index -s noarch -n hdc $(HDC_INDEX)

.PHONY: upload
upload: refresh-index
	@echo Check Upload:
	cd $(HDC_INDEX) && aws s3 sync --exclude='**/.cache/*' --dryrun . s3://wfp-share/hdc/
	@echo "Will do for real in 5 seconds"
	@sleep 5
	cd $(HDC_INDEX) && aws s3 sync --exclude='**/.cache/*' . s3://wfp-share/hdc/