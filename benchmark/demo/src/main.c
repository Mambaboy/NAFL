#include "pwnInit.h"
#include "unit.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

void convert(unsigned char *buf, int size)
{
    unsigned char tmp;
    int i, j;
    for (i = 0; i < size; i++)
    {
        for (j = i; j < size; j++)
        {
            tmp = buf[i + size * j];
            buf[i + size * j] = buf[i * size + j];
            buf[i * size + j] = tmp;
        }
    }
}

int login()
{
    char pwd[0x10];
    char buf[0x10];
    int id;
    FILE *fp = fopen("/dev/urandom", "rb");
    fread(pwd, 12, 1, fp);
    puts("input your password:");
    read_n(buf, 0x10);
    if (!strncmp(buf, pwd, 12))
    {
        puts("welcome , admin!!");
        return 0;
    }
    else
    {
        puts("input your user id:");
        id = read_int();
        if (id > 255 || !id)
        {
            puts("user id is too big or you can't set you to admin!!");
            exit(0);
        }
        return id;
    }
}

int check_root(int id)
{
    if (!(short)id)
    {
        return 1;
    }
    return 0;
}


void run(){

    struct __stack{
        int admin;
        int num;
        int size;
        int id;
        int i, j;
        unsigned char buf[25];
    }stack;

    stack.admin=0;
    stack.num=25;
    stack.size=5;
    
    puts("welcome to my char matrix transposition system :)");
    stack.id = login();
    stack.admin = check_root(stack.id);
    if (stack.admin)
    {
        puts("welcome admin, you can convert unlimited number!!");
    }
    else
    {
        puts("you are only guest, so you can only convert 5x5 matrix");
    }

    if (stack.admin)
    {
        puts("how many numbers do you want to convert?");
        stack.num = read_int();
        stack.size = (int)sqrt(stack.num);
    }

    
    for (stack.i = 0; stack.i < stack.size * stack.size; stack.i++)
    {
        stack.buf[stack.i] = (unsigned char)read_int();
    }

    convert(stack.buf, stack.size);

    puts("convert result:");
    for (stack.i = 0; stack.i < stack.size; stack.i++)
    {
        for (stack.j = 0; stack.j < stack.size; stack.j++)
        {
            printf("%d ", stack.buf[stack.i * stack.size + stack.j]);
        }
        puts("");
    }

}

int main()
{
    PWNINIT
    run();
    return 0;
}
