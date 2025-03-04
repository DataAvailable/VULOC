## About VULOC
VULOC is a vulnerability scanning tool for C/C++ source code. It uses a customized deep learning architecture, combined with high-level abstract features of vulnerability source code and low-level fine-grained features of assembly code, to detect vulnerable functions and accurately locate vulnerable lines.

## How to replicate

### gcc编译

```shell
gcc -g -o test -I ./testcasesupport -D INCLUDEMAIN test.c
gcc -g -o test -I ./testcasesupport -D INCLUDEMAIN -D OMITGOOD test.c
gcc -g -o test -I ./testcasesupport -D INCLUDEMAIN -D OMITBAD test.c
```

### Python调用GDB调试

```shell
gdb.execute('file test')
set logging redirect on
set logging on vulnerable.txt
set logging on non_vulnerable.txt
set pagination off								# 取消GDB信息的分页显示
gdb.execute('disas /m func')
gdb -q -x file.py
```

```python
p = gdb.execute('disas /m ' + file, to_string=True)  # 在python中获取gdb的输出 
print(p)
```

```shell
info functions		# 列出可执行文件的所有函数名称
```

