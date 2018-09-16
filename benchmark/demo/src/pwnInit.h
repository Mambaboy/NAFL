//
// Created by explorer on 17-8-4.
//

#ifndef PROJECT_PWNINIT_H
#define PROJECT_PWNINIT_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void init() {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    alarm(300);
}


#define PWNINIT init();

#endif //PROJECT_PWNINIT_H
