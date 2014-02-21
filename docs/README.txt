This directory contains

* Doxyfile, doxygen_custom.css : Config files for doxygen 
    doxygen will generate the directories html/ and latex/. Do "make clobber" to remove them.

* tutorial/ : Directory containing .tex file to generate PDF version of the documentation.

* images/ : Images used in the documentation
    contains sources (mainly SVG), and PNG/PDF files generated from the sources.
    
* dmp.bib : BibTex file with relevant work. Doesn't work yet with my older doxygen version...

* makefile : To recursively call make in tutorial/
