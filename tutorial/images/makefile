################################################################################
# SVG files cannot be directly included in LaTeX. But pdflatex does accept PDF
# and PNG files. Therefore, this makefile converts SVG into PDF and PNG.
#
# Why both? Because very large SVG files (>1Mb, e.g. a graph with many data
# points) might lead to large PDF files. In this case, the PNG is often smaller
# (at the cost of being a rasterized bitmap image instead of a vector graphic).
#
# In the LaTeX file, the PDF/PNG are included as follows:
#   \includegraphics{filename-svg}      (PDF file generated from SVG)
#   \includegraphics{filename-svg-png}  (PNG file generated from SVG)
#   \includegraphics{filename-png}      (PNG file)
#   \includegraphics{filename-pdf}      (PDF file)


################################################################################
# RELEVANT FILES

# Get list of all SVG files
SOURCE_SVG := $(shell find -name '*.svg')
TARGET_SVG_PDF = $(SOURCE_SVG:.svg=-svg.pdf)
TARGET_SVG_PNG = $(SOURCE_SVG:.svg=-svg.png)

# Get list of all PNG files
SOURCE_PNG := $(shell find -name '*.png')
TARGET_PNG_PNG = $(SOURCE_PNG:.png=-png.png)

# Get list of all PDF files
SOURCE_PDF := $(shell find -name '*.pdf')
TARGET_PDF_PDF = $(SOURCE_PDF:.pdf=-pdf.pdf)

# Add output directory
BIN_DIR = ./
TARGET_SVG_PDF := $(addprefix ${BIN_DIR}/,${TARGET_SVG_PDF})
TARGET_SVG_PNG := $(addprefix ${BIN_DIR}/,${TARGET_SVG_PNG})
TARGET_PNG_PNG := $(addprefix ${BIN_DIR}/,${TARGET_PNG_PNG})
TARGET_PDF_PDF := $(addprefix ${BIN_DIR}/,${TARGET_PDF_PDF})


################################################################################
# TARGETS

all: ${TARGET_SVG_PDF}  ${TARGET_SVG_PNG} #${TARGET_PNG_PNG} ${TARGET_PDF_PDF} 
	@echo "__________________________________________________\nBuilding images"

clean:
	rm ${TARGET_SVG_PDF} ${TARGET_SVG_PNG} #${TARGET_PNG_PNG} ${TARGET_PDF_PDF}

${BIN_DIR}:
	mkdir ${BIN_DIR}

################################################################################
# PATTERN RULES

# Command to convert SVG file into a PNG/PDF file
#   --without-gui         : Don't start inkscape GUI
#   --export-area-drawing : Bounding box tight around the actual drawing
SVG_EXPORT=inkscape --without-gui --export-area-drawing 

#   --export-text-to-path : Convert text to paths before exporting
${BIN_DIR}/%-svg.pdf:%.svg
	@echo "$< -> $@"
	@${SVG_EXPORT} --export-text-to-path --file=$<  --export-pdf=$@ 2> /dev/null > /dev/null

#   --export-use-hints    : Get hints about resolution from filename
#                           If not available: the default resolution is 90 DPI
${BIN_DIR}/%-svg.png:%.svg
	@echo "$< -> $@"
	@${SVG_EXPORT} --export-use-hints --file=$< --export-png=$@ 2> /dev/null > /dev/null

#	${SVG_EXPORT} --export-dpi=30 --file=$< --export-png=$@ 2> /dev/null > /dev/null


# PNG from PNG is easy ;-)
${BIN_DIR}/%-png.png:%.png ${BIN_DIR}
	@echo "$< -> $@"
	@cp $< $@
	
# PDF from PDF is easy ;-)
${BIN_DIR}/%-pdf.pdf:%.pdf ${BIN_DIR}
	@echo "$< -> $@"
	@cp $< $@


