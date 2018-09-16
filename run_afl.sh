if [ ! $# == 2 ]; then
    echo "Usage: $0 engine binary"
    exit
fi

Target=$2
Engine=$1
PWD=`pwd`

if [ "$Engine"x = "afl"x ]; then
	AFL_HOME=$PWD/afl # afl
	OUTPUT=$PWD/output-afl-$Target
	echo "using afl"
elif [ "$Engine"x = "fair"x ]; then
	AFL_HOME=$PWD/afl-rb #aflnb
	OUTPUT=$PWD/output-fair-$Target
	echo "using aflnb"
else
    echo "enging wrong"
    exit
fi

rm -rf $OUTPUT
mkdir -p $OUTPUT

$AFL_HOME/afl-fuzz -i $PWD/seed -o $OUTPUT  ./benchmark/$Target
