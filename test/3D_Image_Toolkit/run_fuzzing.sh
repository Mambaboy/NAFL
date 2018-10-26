if [ ! $# == 1 ]; then
    echo "Usage: $0 engine "
    exit
fi

Target=3D_Image_Toolkit
Engine=$1
PWD=`pwd`

if [ "$Engine"x = "afl"x ]; then
	AFL_HOME=$PWD/../../afl-neuzz # afl
	OUTPUT=/tmp/output-afl-$Target
	echo "using afl"
elif [ "$Engine"x = "fair"x ]; then
	AFL_HOME=$PWD/../../afl-rb #aflnb
	OUTPUT=/tmp/output-fair-$Target
	echo "using fair"
elif [ "$Engine"x = "orig"x ]; then
	AFL_HOME=$PWD/../../afl-orig #aflnb
	OUTPUT=/tmp/output-orig-$Target
	echo "using orig"
else
    echo "enging wrong"
    exit
fi

rm -rf $OUTPUT
mkdir -p $OUTPUT

$AFL_HOME/afl-fuzz -i $PWD/seed -o $OUTPUT  $PWD/$Target 
