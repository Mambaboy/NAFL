Output_dir="./output-fair"
rm -rf $Output_dir
mkdir -p $Output_dir
  ./afl-rb/afl-fuzz -i ./seed -o $Output_dir  -Q ./benchmark/size @@
