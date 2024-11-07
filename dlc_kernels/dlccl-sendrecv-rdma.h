#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"

#ifndef _H_X86_DLCCL_SENDRECV_RDMA_
#define _H_X86_DLCCL_SENDRECV_RDMA_

#include "dlccl-rdma.h"

const int SENDRECV_HANDSHAKE_SYNC = EXCHIP_COMMU_SYNC;
const int SENDRECV_HANDSHAKE_B_SYNC = EXCHIP_COMMU_SYNC_2;
const int SENDRECV_HANDSHAKE_C_SYNC = EXCHIP_COMMU_SYNC_3;
const int HANDSHAKE_REQUIRE = 0x2E0000;
const int HANDSHAKE_DENY = 0xDE0000;
const int HANDSHAKE_ACK = 0xA30000;

inline void handshake_send(int selfChip, int targetChip) {
    rsync_set_done(targetChip, SENDRECV_HANDSHAKE_C_SYNC);
    sync_wait_done(SENDRECV_HANDSHAKE_C_SYNC);

    while (1) {
        rsync_set_done_val(targetChip, SENDRECV_HANDSHAKE_SYNC, HANDSHAKE_REQUIRE | (selfChip & 0xffff));
        dlc_s_delay(2000);
        int rep = sync_read(SENDRECV_HANDSHAKE_B_SYNC);
        if ((rep & 0xff0000) == HANDSHAKE_ACK) {
            break;
        }
    }
    sync_clear(SENDRECV_HANDSHAKE_B_SYNC);
}

inline void handshake_recv(int selfChip, int targetChip) {
    rsync_set_done(targetChip, SENDRECV_HANDSHAKE_C_SYNC);
    sync_wait_done(SENDRECV_HANDSHAKE_C_SYNC);

    while (1) {
        int req = sync_read(SENDRECV_HANDSHAKE_SYNC);
        int chip = req & 0xffff;
        int comm = req & 0xff0000;
        if (comm == HANDSHAKE_REQUIRE && chip == targetChip) {
            rsync_set_done_val(targetChip, SENDRECV_HANDSHAKE_B_SYNC, HANDSHAKE_ACK);
            break;
        }
        dlc_s_delay(2000);
    }
}

#endif
