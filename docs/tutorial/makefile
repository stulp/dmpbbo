TARGETS := tutorial.pdf

all: ${TARGETS}

%.pdf:%.tex
	@echo "__________________________________________________\nBuilding $@"
	pdflatex -interaction batchmode $<
	bibtex $<
	pdflatex -interaction batchmode $<
	pdflatex -interaction batchmode $<
	
clean:
	rm -rf *~ *.aux *.log *.nav *.snm *.out *.toc *.blg

clobber: clean
	rm -f ${TARGETS} *.bbl

