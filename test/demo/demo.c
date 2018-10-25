#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
void test1 (char *buf) {
	int n = 0;
	if(buf[0] == 'b')
        n++;
	if(buf[1]+ buf[0] == 'a')
        n++;
	if(buf[2]+ buf[1] == 'd')
        n++;
	if(buf[3]+ buf[2] == '!')
        n++;
	if(buf[4]+ buf[3] == 'b') 
        n++;
	if(buf[5]+ buf[4] == 'a')
        n++;
	if(buf[6]+ buf[5] == 'd')
        n++;
	if(buf[7]+ buf[6] == '!')
        n++;
	if(n == 8)
        raise(SIGSEGV);
}


void test2 (char *buf) {
	if(buf[0] == 'b') 
		if(buf[1] == 'a')
			if(buf[2] == 'd') 
				if(buf[3] == '!') 
					if(buf[4] == 'b')
						if(buf[5] == 'a')
							if(buf[6] == 'd')
								raise(SIGSEGV);

}

int calculation(char* buff){
	int result=0;
	int i=0;
	for (i = 0; i < 4 ; i++) {
		result += buff[i*4]<<(i*8);
	}
	return result; 
}

int main(int argc, char *argv[]) {
	char buf[500];
	FILE* input = NULL;
	input = fopen(argv[1], "r");
	if (input != 0) {
		fread(&buf, 1,500, input);
		test2(buf);
		fclose(input);
	}
	return 0;
}
