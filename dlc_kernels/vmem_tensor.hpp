#pragma once
#include "../dlc-intrinsics.h"
#include "../typehint.h"



#include "align.h"
#include "math.h"

#define ALIGN32(a) (((a) + 31) & 0xffffffe0)

namespace kernel {

    struct Tensor1D
    {
        // double buffer for flatten SIM_X86::tensor, split into two xys, including padding
        // block_sizebytes: size of one block in bytes, 4096B aligned
        Tensor1D(SIM_X86::DLCMem* mem_info, int block_sizebytes, DLCTensor* input)
        : _idx_prev(0), _init(1), _sync_counter(0), core_id(get_device_id()){
            _block_size512 = block_sizebytes / 4096 * 8;
            // vmem0
            vmem.vmem0 = mem_info->vmem_addr;
            mem_info->vmem_addr = (SIM_X86::tensor)((int)mem_info->vmem_addr + _block_size512 * 4);
            mem_info->vmem_size -= _block_size512 * 512;
            // vmem1
            vmem.vmem1 = mem_info->vmem_addr;
            mem_info->vmem_addr = (SIM_X86::tensor)((int)mem_info->vmem_addr + _block_size512 * 4);
            mem_info->vmem_size -= _block_size512 * 512;

            // two xys
            int hbm_size512 = input->dim0_padded * input->dim1 * input->layout / 512;
            int size_xys0 = hbm_size512 / 16 * 8;    // align to 4096B
            int size_xys1 = hbm_size512 - size_xys0;
            int addr_xys0 = (int)input->address;
            int addr_xys1 = addr_xys0 + size_xys0 * 4;
            _hbm_size512 = core_id == 0 ? size_xys0 : size_xys1;
            _hbm_addr = core_id == 0 ? addr_xys0 : addr_xys1;

            vmem.offset512 = core_id == 0 ? 0 : size_xys0;
            num_blocks = soft_sdiv(_hbm_size512 + _block_size512 - 1, _block_size512);
            vmem.w512 = num_blocks == 1 ? _hbm_size512 : _block_size512;
            vmem.address = vmem.vmem0;
            vmem.idx = 0;
        }

        // prefetch
        constexpr inline void _load_next(int next_idx, SIM_X86::tensor vmem_addr){
            SIM_X86::tensor src_addr = (SIM_X86::tensor)(_hbm_addr + next_idx * _block_size512 * 4);
            SIM_X86::tensor dst_addr = vmem_addr;
            int dma_size = (next_idx == num_blocks - 1 ? _hbm_size512 - next_idx * _block_size512 : _block_size512);
            // Print((char*)"load block: %d\n", next_idx);
            vmem.handle = dlc_dma(src_addr, HBM, dst_addr, VMEM, dma_size * 128, 128, 128, 128, 7);
        }

        constexpr inline void _store_curr(int next_idx, SIM_X86::tensor vmem_addr){
            SIM_X86::tensor src_addr = vmem_addr;
            SIM_X86::tensor dst_addr = (SIM_X86::tensor)(_hbm_addr + next_idx * _block_size512 * 4);
            int dma_size = (next_idx == num_blocks - 1 ? _hbm_size512 - next_idx * _block_size512 : _block_size512);
            // Print((char*)"store block: %d\n", vmem.idx);
            vmem.handle = dlc_dma(src_addr, VMEM, dst_addr, HBM, dma_size * 128, 128, 128, 128, 7);
            _sync_counter += dma_size / 128;
        }

        constexpr inline void _update_vmem(int idx){
            vmem.idx = idx;
            vmem.w512 = idx == num_blocks - 1 ? _hbm_size512 - idx * _block_size512 : _block_size512;
            vmem.address = vmem.address == vmem.vmem0 ? vmem.vmem1 : vmem.vmem0;
            if (idx > 0) vmem.offset512 += _block_size512;
        }

        constexpr inline void _sync_dma(){
            dlc_sync_gte(vmem.handle, _sync_counter);
            dlc_sync(vmem.handle);
            _sync_counter = 0;
        }

        constexpr inline void load(){
            _update_vmem(_idx_prev);
            if (_init){
                _load_next(_idx_prev, vmem.address);
                _init = 0;
            }
            dlc_sync(vmem.handle);
            if (vmem.idx < num_blocks - 1){
                _load_next(vmem.idx + 1, vmem.address == vmem.vmem0 ? vmem.vmem1 : vmem.vmem0);
            }
            _idx_prev = vmem.idx + 1;
        }

        constexpr inline void store(){
            // sync prev, issue curr, switch to next
            _sync_dma();
            _store_curr(vmem.idx, vmem.address);
            _update_vmem(vmem.idx + 1);
            if (vmem.idx >= num_blocks) {
                _sync_dma();
            }
        }

        struct TensorVmem {
            int handle = DONE;
            SIM_X86::tensor  vmem0;
            SIM_X86::tensor  vmem1;
            SIM_X86::tensor  address;         // current valid address, vmem0 or vmem1
            int w512;
            int offset512;          // hbm offset
            int idx;
        } vmem;

        int _idx_prev;
        int _init;
        int _sync_counter;
        int _block_size512;
        int _hbm_size512;
        int _hbm_addr;

        int core_id;
        int num_blocks;             // number of blocks
    };

    enum class SplitType {
        SPLIT_NONE,
        SPLIT_LEFT,
        SPLIT_RIGHT,
        SPLIT_TOP,
        SPLIT_BOTTOM
    };

    template <bool DOUBLEBUFFER>
    struct Tensor2D
    {
        // double buffer for 2D SIM_X86::tensor, support split & transpose
        // t:
        //   SIM_X86::tensor with physical width and height
        // block_w, block_h:
        //   logical block width and height, must be aligned to 128 for fp32, 256 for bf16
        // split:
        //   logical split direction
        // transposed:
        //   whether the SIM_X86::tensor is transposed
        inline Tensor2D(SIM_X86::DLCMem* mem_info, int block_w_, int block_h_, SplitType split, int transposed, DLCTensor* t)
        : _idx_w_prev(0), _idx_h_prev(0), _sync_counter(0), _init(1), _transposed(transposed), _itemsize(t->layout){
            // physical block size
            this->vmem.block_w = transposed ? block_h_ : block_w_;
            this->vmem.block_h = transposed ? block_w_ : block_h_;
            // vmem0, vmem1
            this->vmem.vmem0 = mem_info->vmem_addr;
            mem_info->vmem_addr = (SIM_X86::tensor)((int)mem_info->vmem_addr + vmem.block_w * vmem.block_h * _itemsize / 128);
            mem_info->vmem_size -= vmem.block_w * vmem.block_h * _itemsize;
            if (DOUBLEBUFFER){
                this->vmem.vmem1 = mem_info->vmem_addr;
                mem_info->vmem_addr = (SIM_X86::tensor)((int)mem_info->vmem_addr + vmem.block_w * vmem.block_h * _itemsize / 128);
                mem_info->vmem_size -= vmem.block_w * vmem.block_h * _itemsize;
            } else {
                this->vmem.vmem1 = this->vmem.vmem0;
            }
            this->vmem.address = this->vmem.vmem1;
            // physical block
            int num_blocks_w_full = soft_sdiv(t->dim0_padded + vmem.block_w - 1, vmem.block_w);
            int num_blocks_h_full = soft_sdiv(t->dim1 + vmem.block_h - 1, vmem.block_h);
            if (split == SplitType::SPLIT_NONE){
                num_blocks_w = num_blocks_w_full;
                num_blocks_h = num_blocks_h_full;
                _hbm_addr = (int)t->address;
                _hbm_w = t->dim0;
                _hbm_h = t->dim1;
            } else if ((split == SplitType::SPLIT_LEFT && !transposed) || (split == SplitType::SPLIT_TOP && transposed)){
                num_blocks_w = num_blocks_w_full / 2;
                num_blocks_h = num_blocks_h_full;
                _hbm_addr = (int)t->address;
                _hbm_w = num_blocks_w * vmem.block_w;
                _hbm_h = t->dim1;
            } else if ((split == SplitType::SPLIT_RIGHT && !transposed) || (split == SplitType::SPLIT_BOTTOM && transposed)){
                num_blocks_w = num_blocks_w_full - num_blocks_w_full / 2;
                num_blocks_h = num_blocks_h_full;
                _hbm_addr = (int)t->address + num_blocks_w_full / 2 * vmem.block_w * _itemsize / 128;
                _hbm_w = t->dim0 - num_blocks_w_full / 2 * vmem.block_w;
                _hbm_h = t->dim1;
            } else if ((split == SplitType::SPLIT_TOP && !transposed) || (split == SplitType::SPLIT_LEFT && transposed)){
                num_blocks_w = num_blocks_w_full;
                num_blocks_h = num_blocks_h_full / 2;
                _hbm_addr = (int)t->address;
                _hbm_w = t->dim0;
                _hbm_h = num_blocks_h * vmem.block_h;
            } else if ((split == SplitType::SPLIT_BOTTOM && !transposed) || (split == SplitType::SPLIT_RIGHT && transposed)){
                num_blocks_w = num_blocks_w_full;
                num_blocks_h = num_blocks_h_full - num_blocks_h_full / 2;
                _hbm_addr = (int)t->address + num_blocks_h_full / 2 * vmem.block_h * t->dim0_padded * _itemsize / 128;
                _hbm_w = t->dim0;
                _hbm_h = t->dim1 - num_blocks_h_full / 2 * vmem.block_h;
            }
            this->_hbm_stride = t->dim0_padded * _itemsize / 128;
            this->_update_vmem(0,0);
        }

        constexpr inline void _sync_dma(){
            // Print((char*)"sync gte: %d\n", _sync_counter);
            dlc_sync_gte(vmem.handle, _sync_counter);
            dlc_sync_clear(vmem.handle);
            _sync_counter = 0;
        }

        constexpr inline void _update_vmem(int next_idx_w, int next_idx_h){
            // switch to next vmem
            vmem.w = next_idx_w == num_blocks_w - 1 ? _hbm_w - next_idx_w * vmem.block_w : vmem.block_w;
            vmem.h = next_idx_h == num_blocks_h - 1 ? _hbm_h - next_idx_h * vmem.block_h : vmem.block_h;
            if (DOUBLEBUFFER){
                vmem.address = vmem.address == vmem.vmem0 ? vmem.vmem1 : vmem.vmem0;
            }
        }

        constexpr inline void _load_next(int idx_w, int idx_h, SIM_X86::tensor  vmem_addr){
            int x_offset = idx_w * vmem.block_w * _itemsize / 128;
            int y_offset = idx_h * vmem.block_h * _hbm_stride;
            // int vmem_w = idx_w == num_blocks_w - 1 ? _hbm_w - idx_w * vmem.block_w : vmem.block_w;
            // int vmem_h = idx_h == num_blocks_h - 1 ? _hbm_h - idx_h * vmem.block_h : vmem.block_h;
            // Print((char*)"load block h: %d >>>>>>>>>>>>>>>>>>>>>>\n", idx_h);
            // Print((char*)"load block w: %d\n", idx_w);
            // Print((char*)"vmem.block_h: %d\n", vmem.block_h);
            int curr_vmem_addr = (int)vmem_addr;
            for (int i = 0; i < vmem.block_w * _itemsize; i+=512){
                vmem.handle = dlc_dma(
                    (SIM_X86::tensor)(_hbm_addr + y_offset + x_offset + i / 128), HBM,
                    (SIM_X86::tensor)(curr_vmem_addr + i / 128), VMEM,
                    128*vmem.block_h,   // length
                    _hbm_stride * 32, vmem.block_w * _itemsize / 128 * 32, 128, 7
                );
                _sync_counter += vmem.block_h;
            }
            // Print((char*)"sync_counter: %d\n", _sync_counter);
        }

        constexpr inline void _store_curr(int idx_w, int idx_h, SIM_X86::tensor  vmem_addr){
            int x_offset = idx_w * vmem.block_w * _itemsize / 128;
            int y_offset = idx_h * vmem.block_h * _hbm_stride;
            // int vmem_w = idx_w == num_blocks_w - 1 ? _hbm_w - idx_w * vmem.block_w : vmem.block_w;
            // int vmem_h = idx_h == num_blocks_h - 1 ? _hbm_h - idx_h * vmem.block_h : vmem.block_h;
            // Print((char*)"store block h: %d <<<<<<<<<<<<<<<<<<<<<<<\n", idx_h);
            // Print((char*)"store block w: %d\n", idx_w);
            int curr_vmem_addr = (int)vmem_addr;
            for (int i = 0; i < vmem.block_w * _itemsize; i+=512){
                vmem.handle = dlc_dma(
                    (SIM_X86::tensor)(curr_vmem_addr + i / 128), VMEM,
                    (SIM_X86::tensor)(_hbm_addr + y_offset + x_offset + i / 128), HBM,
                    128*vmem.block_h,   // length
                    vmem.block_w * _itemsize / 128 * 32, _hbm_stride * 32, 128, 7
                );
                _sync_counter += vmem.block_h;
            }
        }

        constexpr inline void load(int next_idx_w_in, int next_idx_h_in){
            int next_idx_w = _transposed ? next_idx_h_in : next_idx_w_in;
            int next_idx_h = _transposed ? next_idx_w_in : next_idx_h_in;
            this->_update_vmem(_idx_w_prev, _idx_h_prev);
            if (DOUBLEBUFFER){
                if (_init){
                    this->_load_next(_idx_w_prev, _idx_h_prev, vmem.address);
                    _init = 0;
                }
                // sync curr block
                this->_sync_dma();
                // load next block
                if (next_idx_w < num_blocks_w && next_idx_h < num_blocks_h){
                    this->_load_next(next_idx_w, next_idx_h, vmem.address == vmem.vmem0 ? vmem.vmem1 : vmem.vmem0);
                }
            } else {
                this->_load_next(_idx_w_prev, _idx_h_prev, vmem.address);
                this->_sync_dma();
            }
            _idx_w_prev = next_idx_w;
            _idx_h_prev = next_idx_h;
        }

        constexpr inline void store(int next_idx_w_in, int next_idx_h_in){
            // sync prev, issue curr, switch to next
            int next_idx_w = _transposed ? next_idx_h_in : next_idx_w_in;
            int next_idx_h = _transposed ? next_idx_w_in : next_idx_h_in;
            if (DOUBLEBUFFER){
                this->_sync_dma();
                this->_store_curr(_idx_w_prev, _idx_h_prev, vmem.address);
                this->_update_vmem(next_idx_w, next_idx_h);
                if (next_idx_w >= num_blocks_w || next_idx_h >= num_blocks_h){
                    this->_sync_dma();
                }
            } else {
                this->_store_curr(_idx_w_prev, _idx_h_prev, vmem.address);
                this->_sync_dma();
                this->_update_vmem(next_idx_w, next_idx_h);
            }
            _idx_w_prev = next_idx_w;
            _idx_h_prev = next_idx_h;
        }

        struct TensorVmem {
            int handle = DONE;
            SIM_X86::tensor  vmem0;
            SIM_X86::tensor  vmem1;
            SIM_X86::tensor  address;         // current valid address, vmem0 or vmem1
            int w;                  // current valid width (unpadded)
            int h;                  // current valid height
            int block_w;            // block width
            int block_h;            // block height
            int idx_w;              // current block index in width
            int idx_h;              // current block index in height
        } vmem;

        int num_blocks_w;
        int num_blocks_h;

        int _idx_w_prev;
        int _idx_h_prev;
        int _sync_counter;
        int _init;
        int _transposed;
        int _hbm_addr;              // unit: 128B
        int _hbm_stride;            // unit: 128B
        int _hbm_w;                 // hbm width, unit: itemsize
        int _hbm_h;                 // hbm height, unit: itemsize
        int _itemsize;
    };
}
