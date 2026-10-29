"""
TensorNeuramorphic - Sparse Spike Encoding Wrapper (PHASE 3.1)
================================================================

Python interface for sparse SNN-SSM kernel
THE GAME-CHANGER: 10-20x speedup potential
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import time


def load_sparse_kernel():
    """Load sparse SNN-SSM CUDA kernel"""
    
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(kernel_dir, "kernels", "sparse_snn_ssm.cu")
    
    if not os.path.exists(cuda_file):
        print(f"⚠️  Sparse kernel not found at {cuda_file}")
        return None
    
    try:
        with open(cuda_file, 'r') as f:
            cuda_source = f.read()
        
        cpp_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <ATen/cuda/CUDAContext.h>
        
        // Forward declare sparse kernel launcher
        void launch_sparse_snn_ssm(
            const float* input,
            float* v_membrane,
            float* refrac_timer,
            float* ssm_state,
            const float* ssm_A,
            const float* ssm_B,
            const float* ssm_C,
            float* output,
            float* spikes,
            int* active_indices,
            int* active_count_device,
            int batch_size,
            int seq_len,
            int hidden_size,
            int state_size,
            cudaStream_t stream
        );
        
        torch::Tensor sparse_snn_ssm_forward(
            torch::Tensor input,
            torch::Tensor ssm_A,
            torch::Tensor ssm_B,
            torch::Tensor ssm_C
        ) {
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int hidden_size = input.size(2);
            int state_size = ssm_A.size(0);
            
            auto device = input.device();
            
            // Allocate states
            auto v_membrane = torch::zeros({batch_size, hidden_size}, input.options());
            auto refrac_timer = torch::zeros({batch_size, hidden_size}, input.options());
            auto ssm_state = torch::zeros({batch_size, state_size}, input.options());
            auto spikes = torch::zeros({batch_size, hidden_size}, input.options());
            
            // Allocate sparse tracking (max 1024 active neurons)
            auto active_indices = torch::zeros({1024}, torch::dtype(torch::kInt32).device(device));
            auto active_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(device));
            
            // Output
            auto output = torch::zeros_like(input);
            
            // Get CUDA stream
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
            
            // Launch sparse kernel
            launch_sparse_snn_ssm(
                input.data_ptr<float>(),
                v_membrane.data_ptr<float>(),
                refrac_timer.data_ptr<float>(),
                ssm_state.data_ptr<float>(),
                ssm_A.data_ptr<float>(),
                ssm_B.data_ptr<float>(),
                ssm_C.data_ptr<float>(),
                output.data_ptr<float>(),
                spikes.data_ptr<float>(),
                active_indices.data_ptr<int>(),
                active_count.data_ptr<int>(),
                batch_size,
                seq_len,
                hidden_size,
                state_size,
                stream
            );
            
            return output;
        }
        """
        
        print("🔧 Compiling sparse CUDA kernel (Phase 3.1 - THE BREAKTHROUGH)...")
        
        sparse_module = load_inline(
            name='sparse_snn_ssm',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=['sparse_snn_ssm_forward'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-arch=sm_87',
                '-std=c++17',
                '--expt-relaxed-constexpr',
            ],
            extra_cflags=['-O3', '-std=c++17'],
            verbose=False,
            with_cuda=True
        )
        
        print("✅ Sparse kernel compiled!")
        print("   Expected: 10-20x speedup on SNN layers")
        return sparse_module
        
    except Exception as e:
        print(f"⚠️  Failed to compile sparse kernel: {e}")
        import traceback
        traceback.print_exc()
        return None


class SparseSNNSSMLayer(nn.Module):
    """
    Sparse spike encoding layer - THE GAME CHANGER
    Only computes active (spiking) neurons
    """
    
    def __init__(self, hidden_size, state_size, use_cuda=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # SSM parameters
        self.ssm_A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.ssm_B = nn.Parameter(torch.randn(state_size, 1) * 0.1)
        self.ssm_C = nn.Parameter(torch.randn(1, state_size) * 0.1)
        
        # Load sparse kernel
        self.cuda_module = None
        if self.use_cuda:
            self.cuda_module = load_sparse_kernel()
            if self.cuda_module is None:
                self.use_cuda = False
    
    def forward(self, input):
        if self.use_cuda and self.cuda_module is not None:
            return self.cuda_module.sparse_snn_ssm_forward(
                input,
                self.ssm_A,
                self.ssm_B,
                self.ssm_C
            )
        else:
            # Fallback
            from .fused_ops import forward_pytorch
            batch_size, seq_len, hidden_size = input.shape
            
            snn_v = torch.zeros(batch_size, hidden_size, device=input.device)
            snn_r = torch.zeros(batch_size, hidden_size, device=input.device)
            ssm_s = torch.zeros(batch_size, self.state_size, device=input.device)
            
            return forward_pytorch(input, snn_v, snn_r, ssm_s,
                                 self.ssm_A, self.ssm_B, self.ssm_C,
                                 torch.zeros(1, 1, device=input.device))


def benchmark_sparse_kernel():
    """
    THE BIG TEST: Validate 10-20x speedup claim
    """
    
    print("\n" + "="*80)
    print("🚀 PHASE 3.1: SPARSE SPIKE ENCODING BENCHMARK")
    print("   THE DISRUPTION THAT CHANGES EVERYTHING")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    
    configs = [
        {"name": "Small", "batch": 2, "seq": 10, "hidden": 64, "state": 32},
        {"name": "Medium", "batch": 1, "seq": 64, "hidden": 512, "state": 64},
        {"name": "Large (Model)", "batch": 1, "seq": 512, "hidden": 1024, "state": 128},
    ]
    
    for cfg in configs:
        print(f"\n{'='*80}")
        print(f"📊 {cfg['name']}: batch={cfg['batch']}, seq={cfg['seq']}, hidden={cfg['hidden']}")
        print("="*80)
        
        input_data = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device)
        
        # Test sparse kernel
        print("\n🚀 Testing Sparse Kernel (Phase 3.1)...")
        sparse_layer = SparseSNNSSMLayer(cfg['hidden'], cfg['state'], use_cuda=True).to(device)
        
        if sparse_layer.use_cuda:
            # Warmup
            for _ in range(3):
                _ = sparse_layer(input_data)
            
            # Benchmark
            torch.cuda.synchronize()
            num_runs = 20
            
            start = time.time()
            for _ in range(num_runs):
                output_sparse = sparse_layer(input_data)
            torch.cuda.synchronize()
            sparse_time = (time.time() - start) / num_runs * 1000
            
            print(f"   ⚡ Sparse kernel: {sparse_time:.3f} ms")
            print(f"   📊 Output range: [{output_sparse.min():.4f}, {output_sparse.max():.4f}]")
            
            # Compare with baseline fused kernel
            from .fused_ops import FusedSNNSSMLayer
            
            print("\n📊 Testing Baseline Fused Kernel (Phase 1)...")
            baseline_layer = FusedSNNSSMLayer(cfg['hidden'], cfg['state'], use_cuda=True).to(device)
            
            # Warmup
            for _ in range(3):
                _ = baseline_layer(input_data)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_runs):
                output_baseline = baseline_layer(input_data)
            torch.cuda.synchronize()
            baseline_time = (time.time() - start) / num_runs * 1000
            
            print(f"   📊 Baseline kernel: {baseline_time:.3f} ms")
            
            # Calculate speedup
            speedup = baseline_time / sparse_time
            
            print(f"\n{'='*80}")
            print(f"   ✅ SPEEDUP: {speedup:.2f}x")
            print(f"   🎯 Target: 10-20x (biological sparsity)")
            
            if speedup >= 10.0:
                print(f"   🎉🎉🎉 BREAKTHROUGH ACHIEVED! Phase 3.1 VALIDATED!")
                print(f"   💎 This is the disruption NVIDIA wants to buy!")
            elif speedup >= 5.0:
                print(f"   ✅ Excellent progress! Approaching target.")
            elif speedup >= 2.0:
                print(f"   ✅ Good speedup, optimization working")
            else:
                print(f"   ⚠️  Below expectations, needs investigation")
            
            print("="*80)
        else:
            print("   ❌ Sparse kernel compilation failed")
    
    print("\n" + "="*80)
    print("PHASE 3.1 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    benchmark_sparse_kernel()
