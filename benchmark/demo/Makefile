CC=../../afl/afl-clang-fast
OUTPUT_DID=bin
SOURCR_DIR=src
CFLAGS=-fno-stack-protector -m32
LD= -z execstack -m32
LD_D=-z execstack -m32
OBJ=main.o unit.o 
VPATH = ${SOURCR_DIR}:${OUTPUT_DID}

.PHONY: clean

all: dir bin

debug: dir bin-dbg

dyn: dir bin-dyn

bin: ${OBJ}
	${CC}  ${OBJ:%.o=${OUTPUT_DID}/%.o} -o ${OUTPUT_DID}/$@ ${LD} -lm
	strip -s ${OUTPUT_DID}/$@

bin-dbg: ${OBJ}
	${CC}  ${OBJ:%.o=${OUTPUT_DID}/%.o} -o ${OUTPUT_DID}/$@ ${LD} -lm

bin-dyn: ${OBJ}
	${CC}  ${OBJ:%.o=${OUTPUT_DID}/%.o} -o ${OUTPUT_DID}/$@ ${LD_D} -lm
	strip -s ${OUTPUT_DID}/$@

dir:
	-mkdir bin

${OBJ}: %.o: %.c
	${CC} -c $< -o ${OUTPUT_DID}/$@ ${CFLAGS}


clean:
	rm bin/bin bin/*.o

