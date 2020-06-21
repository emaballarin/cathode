#!/bin/zsh
SELF_CALLDIR="$(pwd)"

echo "Cleaning..."

mr ./main.aux
rm ./main.bbl
rm ./main.blg
rm ./main.log
rm ./main.pdf
rm ./main.synctex.gz
rm ./main.toc
rm ./main.out

echo "..."

rm ./thesis.aux
rm ./thesis.bbl
rm ./thesis.blg
rm ./thesis.log
rm ./thesis.pdf
rm ./thesis.synctex.gz
rm ./thesis.toc
rm ./thesis.out

echo "..."

rm ./nnet.aux
rm ./nnet.bbl
rm ./nnet.blg
rm ./nnet.log
rm ./nnet.pdf
rm ./nnet.synctex.gz
rm ./nnet.toc
rm ./nnet.out

echo "..."

rm ./slides.aux
rm ./slides.bbl
rm ./slides.blg
rm ./slides.log
rm ./slides.pdf
rm ./slides.synctex.gz
rm ./slides.toc
rm ./slides.nav
rm ./slides.snm
rm ./slides.out

echo "..."

rm ./texput.log

echo "OK!"
echo " "

cd "$SELF_CALLDIR"
