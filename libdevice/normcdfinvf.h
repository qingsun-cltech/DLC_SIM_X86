#ifndef _NORMCDFINVF_H_
#define _NORMCDFINVF_H_

inline float8_128 __dlc_normcdfinvf(float8_128 a)
{
    float8_128 result0;
    asm (
        "{V0@(pr0)  vr10 = mov.u32 %[input0];}"
        "{"
        "V0@(pr0)	vmsk0 = ls.f32 vr10, r46;"
        "V1@(pr0)	vmsk1 = gt.f32 vr10, r49;"
        "}"
        "{"
        "V0@(pr0)	vmsk2 = eq.f32 vr10, r46;"
        "V1@(pr0)	vmsk3 = vor.f32 vmsk0, vmsk1;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 42992;"
        "pseudo@0	@pseudo imm_1 = 15558;"
        "V0@(pr0)	vmsk4 = eq.f32 vr10, r49;"
        "V1@(pr0)	vmsk5 = ls.f32 vr10, r44;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 51905;"
        "pseudo@0	@pseudo imm_1 = 16249;"
        "pseudo@0	@pseudo imm_2 = 32704;"
        "V0@(pr0)	vmsk6 = gt.f32 vr10, r44;"
        "V1@(pr0)	vr0 = mov.u32 r38;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 65408;"
        "pseudo@0	@pseudo imm_1 = 32640;"
        "V0@(pr0)	vr1 = mov.u32 r37;"
        "V1@(pr0)	vr2 = mov.u32 r36;"
        "}"
        "{"
        "V0@(pr0)	vr9 = mov.u32 r46;"
        "V1@(pr0)	vr8 = sel vmsk5 vr9, vr10;"
        "}"
        "{"
        "V0@(pr0)	vr6 = mov.u32 r49;"
        "V1@(pr0)	vr7 = sub.f32 vr6, vr10;"
        "}"
        "{"
        "V0@(pr0)	vr7 = sel vmsk6 vr9, vr7;"
        "V1@(pr0)	vr9 = or.u32 vr7, vr8;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr9;"
        "V1@(pr0)	vr4 = mov.u32 r53;"
        "}"
        "{"
        "MTR@(pr0)      vr3 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = log.f32 vr4;"
        "}"
        "{"
        "MTR@(pr0)      vr4 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr4;"
        "}"
        "{"
        "MTR@(pr0)      vr4 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr4, vr3;"
        "}"
        "{"
        "V0@(pr0)	vr3 = mul.f32 vr3, r59;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rsqrt.f32 vr3;"
        "V1@(pr0)	vmsk7 = eq.s32 vr3, r47;"
        "}"
        "{"
        "MTR@(pr0)      vr4 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr4;"
        "}"
        "{"
        "MTR@(pr0)      vr4 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr3 = sel vmsk1 vr4, r47;"
        "V1@(pr0)	vr11 = sub.f32 vr10, r50;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 6253;"
        "pseudo@0	@pseudo imm_1 = 48127;"
        "pseudo@0	@pseudo imm_2 = 4390;"
        "pseudo@0	@pseudo imm_3 = 48805;"
        "V0@(pr0)	vr4 = mul.f32 vr3, r44;"
        "V1@(pr0)	vr4 = add.f32 vr4, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 42502;"
        "pseudo@0	@pseudo imm_3 = 49177;"
        "V0@(pr0)	vr4 = mul.f32 vr3, vr4;"
        "V1@(pr0)	vr4 = add.f32 vr4, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 11985;"
        "pseudo@0	@pseudo imm_3 = 49187;"
        "V0@(pr0)	vr4 = mul.f32 vr3, vr4;"
        "V1@(pr0)	vr4 = add.f32 vr4, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 64832;"
        "pseudo@0	@pseudo imm_3 = 16523;"
        "V0@(pr0)	vr4 = mul.f32 vr3, vr4;"
        "V1@(pr0)	vr4 = add.f32 vr4, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 2785;"
        "pseudo@0	@pseudo imm_3 = 16444;"
        "V0@(pr0)	vr4 = mul.f32 vr3, vr4;"
        "V1@(pr0)	vr4 = add.f32 vr4, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 5827;"
        "pseudo@0	@pseudo imm_1 = 15359;"
        "pseudo@0	@pseudo imm_2 = 6761;"
        "pseudo@0	@pseudo imm_3 = 16037;"
        "V0@(pr0)	vr5 = mul.f32 vr3, r44;"
        "V1@(pr0)	vr5 = add.f32 vr5, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 32020;"
        "pseudo@0	@pseudo imm_3 = 16412;"
        "V0@(pr0)	vr5 = mul.f32 vr3, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 18491;"
        "pseudo@0	@pseudo imm_3 = 16496;"
        "V0@(pr0)	vr5 = mul.f32 vr3, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r45;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr3, vr5;"
        "V1@(pr0)	vr5 = add.f32 vr5, r49;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr5;"
        "}"
        "{"
        "MTR@(pr0)      vr5 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr4 = mul.f32 vr4, vr5;"
        "}"
        "{"
        "V0@(pr0)	vr5 = mul.f32 vr4, r57;"
        "}"
        "{"
        "V0@(pr0)	vr6 = mul.f32 vr11, vr11;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 51598;"
        "pseudo@0	@pseudo imm_1 = 49694;"
        "pseudo@0	@pseudo imm_2 = 62004;"
        "pseudo@0	@pseudo imm_3 = 17244;"
        "V0@(pr0)	vr7 = mul.f32 vr6, r44;"
        "V1@(pr0)	vr7 = add.f32 vr7, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 63193;"
        "pseudo@0	@pseudo imm_3 = 50057;"
        "V0@(pr0)	vr7 = mul.f32 vr6, vr7;"
        "V1@(pr0)	vr7 = add.f32 vr7, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 23446;"
        "pseudo@0	@pseudo imm_3 = 17162;"
        "V0@(pr0)	vr7 = mul.f32 vr6, vr7;"
        "V1@(pr0)	vr7 = add.f32 vr7, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 20866;"
        "pseudo@0	@pseudo imm_3 = 49653;"
        "V0@(pr0)	vr7 = mul.f32 vr6, vr7;"
        "V1@(pr0)	vr7 = add.f32 vr7, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 27801;"
        "pseudo@0	@pseudo imm_3 = 16416;"
        "V0@(pr0)	vr7 = mul.f32 vr6, vr7;"
        "V1@(pr0)	vr7 = add.f32 vr7, r45;"
        "}"
        "{"
        "V0@(pr0)	vr7 = mul.f32 vr11, vr7;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_0 = 59270;"
        "pseudo@0	@pseudo imm_1 = 49753;"
        "pseudo@0	@pseudo imm_2 = 38393;"
        "pseudo@0	@pseudo imm_3 = 17185;"
        "V0@(pr0)	vr8 = mul.f32 vr6, r44;"
        "V1@(pr0)	vr8 = add.f32 vr8, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 45808;"
        "pseudo@0	@pseudo imm_3 = 49947;"
        "V0@(pr0)	vr8 = mul.f32 vr6, vr8;"
        "V1@(pr0)	vr8 = add.f32 vr8, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 39494;"
        "pseudo@0	@pseudo imm_3 = 17029;"
        "V0@(pr0)	vr8 = mul.f32 vr6, vr8;"
        "V1@(pr0)	vr8 = add.f32 vr8, r45;"
        "}"
        "{"
        "pseudo@0	@pseudo imm_2 = 32172;"
        "pseudo@0	@pseudo imm_3 = 49492;"
        "V0@(pr0)	vr8 = mul.f32 vr6, vr8;"
        "V1@(pr0)	vr8 = add.f32 vr8, r45;"
        "}"
        "{"
        "V0@(pr0)	vr8 = mul.f32 vr6, vr8;"
        "V1@(pr0)	vr8 = add.f32 vr8, r49;"
        "}"
        "{"
        "V0@(pr0)	(urf) = rcp.f32 vr8;"
        "}"
        "{"
        "MTR@(pr0)      vr8 = pop urf;"
        "}"
        "{"
        "V0@(pr0)	vr7 = mul.f32 vr7, vr8;"
        "V1@(pr0)	vr3 = mov.u32 r46;"
        "}"
        "{"
        "V0@(pr0)	vr8 = sel vmsk3 vr3, vr0;"
        "V1@(pr0)	vr9 = sel vmsk2 vr3, vr2;"
        "}"
        "{"
        "V0@(pr0)	vr10 = sel vmsk4 vr3, vr1;"
        "V1@(pr0)	vr11 = sel vmsk5 vr3, vr4;"
        "}"
        "{"
        "V0@(pr0)	vr6 = sel vmsk6 vr3, vr5;"
        "V1@(pr0)	vr8 = or.u32 vr8, vr9;"
        "}"
        "{"
        "V0@(pr0)	vr8 = or.u32 vr8, vr10;"
        "V1@(pr0)	vr8 = or.u32 vr8, vr11;"
        "}"
        "{"
        "V0@(pr0)	vr8 = or.u32 vr8, vr6;"
        "V1@(pr0)	vmsk0 = eq.f32 vr8, r46;"
        "}"
        "{"
        "V0@(pr0)	%[res0] = sel vmsk0 vr8, vr7;"
        "}"
        ""
        : [res0] "=x" (result0)
        : [input0] "x" (a)
        :"vr2", "vr1", "vr4", "vr5", "vr0", "vr3", "vr7", "vr11", "vr8", "vr10", "vr6", "vr9", "vmsk3", "vmsk0", "vmsk5", "vmsk4", "vmsk7", "vmsk1", "vmsk2", "vmsk6"
        );
    return result0;
}

#endif // _NORMCDFINVF_H_
