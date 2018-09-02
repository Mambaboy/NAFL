Output_dir="./output-afl"
rm -rf $Output_dir
mkdir -p $Output_dir
  ./afl/afl-fuzz -i ./seed -o $Output_dir  -Q ./benchmark/size @@
