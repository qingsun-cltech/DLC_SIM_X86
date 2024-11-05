#ifndef _POPC_H_
#define _POPC_H_

inline int8_128 __dlc_popc(int8_128 a)
{
    int8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	%[res0] = cntone vr10;"
        "}"
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr10"
        );
    return result0;
}

#endif // _POPC_H_
