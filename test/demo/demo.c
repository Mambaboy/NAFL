#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
// void test1 (char *buf) {
// int n = 0;
// if(buf[0] == 'b') n++;
// if(buf[1] == 'a') n++;
// if(buf[2] == 'd') n++;
// if(buf[3] == '!') n++;

// if(n == 4)
//  	{
// 	raise(SIGSEGV);
// 	}
// }

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

int main(int argc, char *argv[]) {
	char buf[500];
	FILE* input = NULL;
	input = fopen(argv[1], "r");
	if (input != 0) {
		fscanf(input, "%128c", &buf);
		test2(buf);
		fclose(input);
	}
	return 0;
}
