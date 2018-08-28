rm -rf /tmp/afl
mkdir -p /tmp/afl
./afl/afl-fuzz -i ./seed -o /tmp/afl  -Q ./benchmark/readelf -a @@
