#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef _H_X86_DLCCL_D2D_
#define _H_X86_DLCCL_D2D_

#include "math.h"
// #include "typehint.h"
#include "../ops/sync.h"

#ifndef SEND_SYNCFLAG
#define SEND_SYNCFLAG 324
#endif

#ifndef COPYSEND_SYNCFLAG
#define COPYSEND_SYNCFLAG 325
#endif

#ifndef RECVCOPYSEND_SYNCFLAG
#define RECVCOPYSEND_SYNCFLAG 326
#endif

#ifndef RECVREDUCESEND_SYNCFLAG
#define RECVREDUCESEND_SYNCFLAG 327
#endif

#ifndef RECVREDUCECOPYSEND_SYNCFLAG
#define RECVREDUCECOPYSEND_SYNCFLAG 328
#endif

#ifndef BARRIER_SYNCFLAG
#define BARRIER_SYNCFLAG 360
#endif

#ifndef CHANNEL_SYNCFLAG_BASE
// use base, base + 4, base + 8, base + 12
#define CHANNEL_SYNCFLAG_BASE 384
#endif

#define INTR_ARG_ADDR 1047040

enum {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
};

#define MAKECHIPID(CHIP, XYS) (((CHIP) << 4) | ((XYS) + 2))

#define CORE_FXCHBM 1
#define CORE_XYS0 2
#define CORE_XYS1 3
#define MEM_H_X86BM 0
#define MEM_CMEM 2
#define MEM_VMEM 0
#define MEM_SMEM 1
#define MEM_IMEM 2
#define DMA_TYPE_LOCAL 0
#define DMA_TYPE_REMOTE_UNICAST 2
#define DMA_TYPE_REMOTE_MULTICAST 3

inline int MakeDmaHeader(int trace_en, int dst_opcode, int dst_coreid, int dst_memid, int src_opcode,
                         int src_codeid, int src_memid, int dmatype, int dst_id) {
    return (dst_id >> 4 /*dst_id contain xysid, remove it*/) | (dmatype << 14) | (src_memid << 16) |
           (src_codeid << 18) | (src_opcode << 21) | (dst_memid << 24) | (dst_coreid << 26) |
           (dst_opcode << 29) | (trace_en << 31);
}

inline int MakeDstSyncFlag(int sync0_core, int sync0_flag, int sync1_core, int sync1_flag) {
    return sync0_flag | ((sync0_core + 2) << 13) | (sync1_flag << 16) | ((sync1_core + 2) << 29);
}

inline int MakeSrcSyncFlag(int sync_core, int sync_flag) { return sync_flag | ((sync_core + 2) << 13); }

inline int GetCoreIdFromChipId(int chipid) { return (chipid & 0b1111) - 2; }

inline void sysTransHbm(uint targetChipId, SIM_X86::tensor srcAddr, SIM_X86::tensor dstAddr, uint len, int flag) {
    volatile int *callArg = (int *)(INTR_ARG_ADDR);
    callArg[0] = targetChipId >> 4;
    callArg[1] = 0;
    callArg[2] = (uint)srcAddr << 7;
    callArg[3] = (uint)srcAddr >> 25;
    callArg[4] = (uint)dstAddr << 7;
    callArg[5] = (uint)dstAddr >> 25;
    callArg[6] = 4 * len;
    callArg[7] = 0;
    callArg[8] = flag;

    asm("{ pseudo@0 @pseudo imm_2 = 2; MISC@(pr0) Nah = intrpt 4; }"
        "{ S0@(pr0) Nah = halt; }");
}

inline void d2d_h2h(SIM_X86::tensor sAddr, int sFlag, int dChip, SIM_X86::tensor dAddr, int dFlag, int len) {
    sysTransHbm(dChip, sAddr, dAddr, len, dFlag);
}

inline void dma_h2h(SIM_X86::tensor src, SIM_X86::tensor dst, int len) {
    int cmemlen = 32 * 1024 * 1024 / 4; // 32MB
    for (int i = 0; i < len; i+= cmemlen) {
        int curlen = min(len - i, cmemlen);
        dlc_sync(dlc_dma(src + (uint)i / 32, HBM, (SIM_X86::tensor)0, CMEM, curlen, 128, 128, 128, 7));
        dlc_sync(dlc_dma((SIM_X86::tensor)0, CMEM, dst + (uint)i / 32, HBM, curlen, 128, 128, 128, 7));
    }
}

inline void rsync_set_done(int chipid, int syncflag) {
    d2d_h2h((SIM_X86::tensor)(0xE00000), 0, chipid, (SIM_X86::tensor)(0xE00000), MakeSrcSyncFlag(0, syncflag), 128);
}

/*
```
State of Tx/Rx

    SELF             NEXT
    TX CLR(OK) ----- RX CLR(NO): Write Ready
    ↓ TX SEND BEGIN  ↑ RX RECV END
    TX SET(NO) ----- RX CLR(NO): Sending Recving
    ↓ TX SEND END    ↑ RX RECV BEGIN
    TX SET(NO) ----- RX SET(OK): Read Ready

    TX CLR(OK) ----- RX SET(OK): Illegal 
```
Tx use CLR for ok so uninited state is just Write Ready
Rx must use SET for ok because dma only can set done
*/
struct Channel {
    SIM_X86::tensor rx;
    SIM_X86::tensor tx;
    int rxSw;
    int txSw;
    int bufSize;

    /*

        prev card: rxReadSyncFlag
                   txWriteSyncFlag <-----+
                                         |
        self card: rxReadSyncFlag        |
                   rxWriteSyncFlag +-----+
                   txWriteSyncFlag
                   txReadSyncFlag +------+
                                         |
        next card: rxReadSyncFlag <------+
                   txWriteSyncFlag
    */

    // own, done for rx ready for read; this same with prev chip's txReadSyncFlag
    // prev chip would set done after send
    // dont't change byself
    int rxReadSyncFlag[2];

    // own, clear for tx ready for write; this same with next chip's rxWriteSyncFlag
    // next chip would set clear after recv
    // dont't change byself
    int txWriteSyncFlag[2];

    // remote in next chip, set done for next chip that tx is send finish
    // this same with rxReadSyncFlag
    // done tell next chip send finish, and ready for recv
    // clear tell next chip is sending
    int txReadSyncFlag[2];

    // remote in prev chip, set clear for next chip that rx is recv finish
    // this same with txWriteSyncFlag
    // clear tell prev chip recv finish, and ready for send
    // done tell prev chip is recving
    int rxWriteSyncFlag[2];

    // prev chip
    int rxChipid;
    // next chip
    int txChipid;
};

inline void initChannel(struct Channel *self, SIM_X86::tensor rx, SIM_X86::tensor tx, int channelSize, int prevChipid,
                        int nextChipid) {
    self->rx = rx;
    self->tx = tx;
    self->rxSw = 0;
    self->txSw = 0;
    self->bufSize = roundDown128(channelSize / 2);
    self->rxChipid = prevChipid;
    self->txChipid = nextChipid;
    self->txWriteSyncFlag[0] = self->rxWriteSyncFlag[0] = CHANNEL_SYNCFLAG_BASE;
    self->txWriteSyncFlag[1] = self->rxWriteSyncFlag[1] = CHANNEL_SYNCFLAG_BASE + 4;
    self->rxReadSyncFlag[0] = self->txReadSyncFlag[0] = CHANNEL_SYNCFLAG_BASE + 8;
    self->rxReadSyncFlag[1] = self->txReadSyncFlag[1] = CHANNEL_SYNCFLAG_BASE + 12;

    sync_set_done(self->txWriteSyncFlag[0]);
    sync_set_done(self->txWriteSyncFlag[1]);
}

inline void clearChannel(struct Channel *self) {
    // sync_clear(self->rxReadSyncFlag[0]);
    // sync_clear(self->rxReadSyncFlag[1]);
    sync_clear(self->txWriteSyncFlag[0]);
    sync_clear(self->txWriteSyncFlag[1]);
}

/*
```
    PREV             SELF
    TX XXX(??) ----- RX SET(OK): Read Ready / Illegal
                     ↓ RX RECV BEGIN 
    TX XXX(??) ----- RX CLR(NO): Sending Recving / Write Ready
```
*/
inline SIM_X86::tensor rxBegin(struct Channel *self) {
    sync_wait_done(self->rxReadSyncFlag[self->rxSw]);
    sync_clear(self->rxReadSyncFlag[self->rxSw]);
    // rsync_set_done(self->rxChipid, self->rxWriteSyncFlag[self->rxSw]);
    return self->rx + (self->rxSw * self->bufSize) / 32;
}

/*
```
    PREV             SELF
    TX XXX(??) ----- RX CLR(NO): Sending Recving / Write Ready
    ↓ RX RECV END 
    TX CLR(OK) ----- RX CLR(NO): Write Ready
```
*/
inline void rxEnd(struct Channel *self) {
    rsync_set_done(self->rxChipid, self->rxWriteSyncFlag[self->rxSw]);
    self->rxSw = 1 - self->rxSw;
}

/*
```
    SELF             NEXT
    TX CLR(OK) ----- RX XXX(??): Write Ready / Illegal
    ↓ TX SEND BEGIN 
    TX SET(NO) ----- RX XXX(??): Sending Recving / Read Ready
```
*/
inline SIM_X86::tensor txBegin(struct Channel *self) {
    sync_wait_done(self->txWriteSyncFlag[self->txSw]);
    sync_clear(self->txWriteSyncFlag[self->txSw]);
    // rsync_clear(self->txChipid, self->txReadSyncFlag[self->txSw]);
    return self->tx + (self->txSw * self->bufSize) / 32;
}

/*
```
    SELF             NEXT
    TX SET(NO) ----- RX XXX(??): Sending Recving / Read Ready
                     ↓ TX SEND END
    TX SET(NO) ----- RX SET(OK): Read Ready
```
*/
inline void txEnd(struct Channel *self) {
    // rsync_set_done(self->txChipid, self->txReadSyncFlag[self->txSw]);
    self->txSw = 1 - self->txSw;
}

/*
    generate dst syncflag used for dma
*/
inline int txDstSyncFlag(struct Channel *self) {
    return MakeDstSyncFlag(GetCoreIdFromChipId(self->txChipid), self->txReadSyncFlag[self->txSw], -2, 0);
}

inline void send(struct Channel *channel, SIM_X86::tensor selfIn, int len) {
    SIM_X86::tensor tx = txBegin(channel);

    // sync_clear(SEND_SYNCFLAG);
    d2d_h2h(selfIn, MakeSrcSyncFlag(0, SEND_SYNCFLAG), channel->txChipid, tx, txDstSyncFlag(channel), len);
    // sync_wait_done(SEND_SYNCFLAG);
    // sync_clear(SEND_SYNCFLAG);

    txEnd(channel);
}

// TODO: directSend is just call send now
inline void directSend(struct Channel *channel, SIM_X86::tensor selfIn, int len) { send(channel, selfIn, len); }

inline void recv(struct Channel *channel, SIM_X86::tensor selfOut, int len) {
    SIM_X86::tensor rx = rxBegin(channel);

    dma_h2h(rx, selfOut, len);

    rxEnd(channel);
}

// TODO: directRecv is just call recv now
inline void directRecv(struct Channel *channel, SIM_X86::tensor selfOut, int len) { recv(channel, selfOut, len); }

inline void copySend(struct Channel *channel, SIM_X86::tensor selfIn, SIM_X86::tensor selfOut, int len) {
    SIM_X86::tensor tx = txBegin(channel);

    // sync_clear(COPYSEND_SYNCFLAG);
    d2d_h2h(selfIn, MakeSrcSyncFlag(0, COPYSEND_SYNCFLAG), channel->txChipid, tx, txDstSyncFlag(channel),
             len);
    dma_h2h(selfIn, selfOut, len);
    // sync_wait_done(COPYSEND_SYNCFLAG);
    // sync_clear(COPYSEND_SYNCFLAG);

    txEnd(channel);
}

// TODO: directCopySend is just call copySend now
inline void directCopySend(struct Channel *channel, SIM_X86::tensor selfIn, SIM_X86::tensor selfOut, int len) {
    copySend(channel, selfIn, selfOut, len);
}

inline void recvCopySend(struct Channel *channel, SIM_X86::tensor selfOut, int len) {
    SIM_X86::tensor rx = rxBegin(channel);
    SIM_X86::tensor tx = txBegin(channel);

    dma_h2h(rx, selfOut, len);
    // sync_clear(RECVCOPYSEND_SYNCFLAG);
    d2d_h2h(rx, MakeSrcSyncFlag(0, RECVCOPYSEND_SYNCFLAG), channel->txChipid, tx, txDstSyncFlag(channel),
             len);
    // sync_wait_done(RECVCOPYSEND_SYNCFLAG);
    // sync_clear(RECVCOPYSEND_SYNCFLAG);

    rxEnd(channel);
    txEnd(channel);
}

// TOOD: directRecvCopySend is just call recvCopySend now
inline void directRecvCopySend(struct Channel *channel, SIM_X86::tensor selfOut, int len) {
    recvCopySend(channel, selfOut, len);
}

typedef float8_128 (*redOp_t)(float8_128, float8_128);

inline void recvReduceCopy(struct Channel *channel, SIM_X86::tensor selfIn, SIM_X86::tensor selfOut, int len, redOp_t redOp,
                           SIM_X86::tensor vmemBuf, int vmemBufLen, bool redOpAvg, int AvgN, bool firstReduce) {
    SIM_X86::tensor rx = rxBegin(channel);

    // OPTME
    int bufLen = roundDown128(vmemBufLen / 2);
    SIM_X86::tensor bufA = vmemBuf;
    SIM_X86::tensor bufB = vmemBuf + bufLen / 32;
    for (int i = 0; i < len; i += bufLen) {
        int curLen = min(len - i, bufLen);
        int syncL1 = dlc_dma(rx + i / 32, HBM, bufA, VMEM, curLen, 128, 128, 128, 7);
        int syncL2 = dlc_dma(selfIn + i / 32, HBM, bufB, VMEM, curLen, 128, 128, 128, 7);
        dlc_sync(syncL1);
        dlc_sync(syncL2);
        for (int j = 0; j < curLen; j += 1024) {
            float8_128 a = v_f32_ld_tnsr_b(j / 32, bufA);
            float8_128 b = v_f32_ld_tnsr_b(j / 32, bufB);
            if (redOpAvg) {
                if (firstReduce) {
                    a = v_f32_mul_b(a, v_f32_rcp_b(v_u32_move_f(AvgN)));
                }
                b = v_f32_mul_b(b, v_f32_rcp_b(v_u32_move_f(AvgN)));
            }
            float8_128 c = redOp(a, b);
            v_f32_st_tnsr_b(j / 32, bufA, c);
        }
        int syncS2 = dlc_dma(bufA, VMEM, selfOut + i / 32, HBM, curLen, 128, 128, 128, 7);
        dlc_sync(syncS2);
    }

    rxEnd(channel);
}

inline void recvReduceSend(struct Channel *channel, SIM_X86::tensor selfIn, int len, redOp_t redOp, SIM_X86::tensor vmemBuf,
                           int vmemBufLen, bool redOpAvg, int AvgN, bool firstReduce) {
    SIM_X86::tensor rx = rxBegin(channel);
    SIM_X86::tensor tx = txBegin(channel);

    // OPTME
    int bufLen = roundDown128(vmemBufLen / 2);
    SIM_X86::tensor bufA = vmemBuf;
    SIM_X86::tensor bufB = vmemBuf + bufLen / 32;
    for (int i = 0; i < len; i += bufLen) {
        int curLen = min(len - i, bufLen);
        int syncL1 = dlc_dma(rx + i / 32, HBM, bufA, VMEM, curLen, 128, 128, 128, 7);
        int syncL2 = dlc_dma(selfIn + i / 32, HBM, bufB, VMEM, curLen, 128, 128, 128, 7);
        dlc_sync(syncL1);
        dlc_sync(syncL2);
        for (int j = 0; j < curLen; j += 1024) {
            float8_128 a = v_f32_ld_tnsr_b(j / 32, bufA);
            float8_128 b = v_f32_ld_tnsr_b(j / 32, bufB);
            if (redOpAvg) {
                if (firstReduce) {
                    a = v_f32_mul_b(a, v_f32_rcp_b(v_u32_move_f(AvgN)));
                }
                b = v_f32_mul_b(b, v_f32_rcp_b(v_u32_move_f(AvgN)));
            }
            float8_128 c = redOp(a, b);
            v_f32_st_tnsr_b(j / 32, bufA, c);
        }

        int syncS = dlc_dma(bufA, VMEM, rx + i / 32, HBM, curLen, 128, 128, 128, 7);
        dlc_sync(syncS);
    }

    // sync_clear(RECVREDUCESEND_SYNCFLAG);
    d2d_h2h(rx, MakeSrcSyncFlag(0, RECVREDUCESEND_SYNCFLAG), channel->txChipid, tx, txDstSyncFlag(channel),
             len);
    // sync_wait_done(RECVREDUCESEND_SYNCFLAG);
    // sync_clear(RECVREDUCESEND_SYNCFLAG);

    rxEnd(channel);
    txEnd(channel);
}

inline void directRecvReduceCopySend(struct Channel *channel, SIM_X86::tensor selfIn, SIM_X86::tensor selfOut, int len,
                                     redOp_t redOp, SIM_X86::tensor vmemBuf, int vmemBufLen, bool redOpAvg, int AvgN,
                                     bool firstReduce) {
    SIM_X86::tensor rx = rxBegin(channel);
    SIM_X86::tensor tx = txBegin(channel);

    // OPTME
    int bufLen = roundDown128(vmemBufLen / 2);
    SIM_X86::tensor bufA = vmemBuf;
    SIM_X86::tensor bufB = vmemBuf + bufLen / 32;
    for (int i = 0; i < len; i += bufLen) {
        int curLen = min(len - i, bufLen);
        int syncL1 = dlc_dma(rx + i / 32, HBM, bufA, VMEM, curLen, 128, 128, 128, 7);
        int syncL2 = dlc_dma(selfIn + i / 32, HBM, bufB, VMEM, curLen, 128, 128, 128, 7);
        dlc_sync(syncL1);
        dlc_sync(syncL2);
        for (int j = 0; j < curLen; j += 1024) {
            float8_128 a = v_f32_ld_tnsr_b(j / 32, bufA);
            float8_128 b = v_f32_ld_tnsr_b(j / 32, bufB);
            if (redOpAvg) {
                if (firstReduce) {
                    a = v_f32_mul_b(a, v_f32_rcp_b(v_u32_move_f(AvgN)));
                }
                b = v_f32_mul_b(b, v_f32_rcp_b(v_u32_move_f(AvgN)));
            }
            float8_128 c = redOp(a, b);
            v_f32_st_tnsr_b(j / 32, bufA, c);
        }
        int syncS2 = dlc_dma(bufA, VMEM, selfOut + i / 32, HBM, curLen, 128, 128, 128, 7);
        dlc_sync(syncS2);
    }

    // sync_clear(RECVREDUCECOPYSEND_SYNCFLAG);
    d2d_h2h(selfOut, MakeSrcSyncFlag(0, RECVREDUCECOPYSEND_SYNCFLAG), channel->txChipid, tx,
             txDstSyncFlag(channel), len);
    // sync_wait_done(RECVREDUCECOPYSEND_SYNCFLAG);
    // sync_clear(RECVREDUCECOPYSEND_SYNCFLAG);

    rxEnd(channel);
    txEnd(channel);
}

#endif