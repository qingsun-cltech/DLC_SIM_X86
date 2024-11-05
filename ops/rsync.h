#ifndef __RSYNC_OPS_H__
#define __RSYNC_OPS_H__

#include "typehint.h"

inline void rsync_inc(int chipid, int syncflag) {
    int target = (chipid << 14) | (syncflag / 4);
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsyncinc 4; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline void rsync_set_val(int chipid, int syncflag, int val) {
    int target = (chipid << 14) | (syncflag / 4);
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) 1 = rmtsetsync 2; }"
                 :
                 : [flag] "r"(target), [val] "r"(val)
                 : "r0", "r1");
}

inline void rsync_clear(int chipid, int syncflag) {
    int target = (chipid << 14) | (syncflag / 4);
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsetsync.clear 0; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline void rsync_set_done(int chipid, int syncflag) {
    int target = (chipid << 14) | (syncflag / 4);
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = rmtsetsync.done 0; }"
                 :
                 : [flag] "r"(target)
                 : "r0");
}

inline void rsync_set_done_val(int chipid, int syncflag, int val) {
    int target = (chipid << 14) | (syncflag / 4);
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val];"
                 "  MISC@(pr0) 1 = rmtsetsync.done 2; }"
                 :
                 : [flag] "r"(target), [val] "r"(val)
                 : "r0", "r1");
}

#endif
