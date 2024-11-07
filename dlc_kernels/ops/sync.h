#pragma once
#include "../../dlc-intrinsics.h"
#include "../../typehint.h"

#ifndef __SYNC_OPS_H_X86__
#define __SYNC_OPS_H_X86__



inline void sync_inc(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = syncinc 4; }"
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

inline void sync_inc_val(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) 1 = syncinc 2; }"
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}

inline void sync_set_val(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  pseudo@0 @pseudo imm_2 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) 1 = setsync 2; }"
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}

inline void sync_clear(int syncflag) {
    // set clear and 0
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = setsync.clear 0; }"
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

inline void sync_set_done(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) 1 = setsync.done 0; }"
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

// sync access

inline int sync_read(int syncflag) {
    int ret;
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) %[ret] = pop;  "
                 "  MISC@(pr0) vtsf = readsync 1; }"
                 : [ret] "=r"(ret)
                 : [flag] "r"(syncflag)
                 : "r0");
    return ret;
}

// sync wait

inline void sync_wait_ge(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) Nah = wait.gte 1, 2; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}


inline void sync_wait_eq(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) Nah = wait.eq 1, 2; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}

inline void sync_wait_ls(int syncflag, int val) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  pseudo@0 @pseudo vs_imm1 = 1; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  S1@(pr0) r1 = mov.u32 %[val]; "
                 "  MISC@(pr0) Nah = wait.ls 1, 2; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag), [val] "r"(val)
                 : "r0", "r1");
}

inline void sync_wait_done(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) Nah = wait.done 1, 0; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

inline void sync_wait_clear(int syncflag) {
    asm volatile("{ pseudo@0 @pseudo vs_imm0 = 0; "
                 "  S0@(pr0) r0 = mov.u32 %[flag]; "
                 "  MISC@(pr0) Nah = wait.undone 1, 0; }"
                 "{ S0@(pr0) Nah = fence; } "
                 :
                 : [flag] "r"(syncflag)
                 : "r0");
}

#endif
