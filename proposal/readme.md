### Using Latex
If you want to compile it locally it's possible to either use an editor or compile .tex files from the command line. `http://www.texstudio.org/` is good editor for windows and has can compile/preview built in. Otherwise, you can install LateX as described below. 

You can also use sharelatex to compile it online. `https://www.sharelatex.com/project` 
### Install Latex
download latex for windows linux or mac at
```
https://miktex.org/download
```
### Compile proposal.tex
```
$ pdflatex proposal.tex
$ bibtex proposal
$ pdflatex proposal.tex
$ pdflatex proposal.tex
```

After the first compile, you will be able to see any updates by running `pdflatex proposal.tex` once. It's neccessary to 'bootstrap' the bibtex dependency to some extent with the multiple compiles at first. 
