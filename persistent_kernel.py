"""
TensorNeuramorphic - Persistent Kernel (PHASE 2.1)
===================================================

BREAKTHROUGH: Single kernel launch processes entire sequence
Eliminates host-device synchronization (3-5x expected speedup)
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import time


def load_persistent_kernel():
    """Load the persistent SNN-SSM CUDA kernel"""
    
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(kernel_dir, "kernels", "fused_snn_ssm_persistent.cu")
    
    if not os.path.exists(cuda_file):
        print(f"⚠️  Persistent kernel not found at {cuda_file}")
        return None
    
    try:
        # Read CUDA source
        with open(cuda_file, 'r') as f:
            cuda_source = f.read()
        
        # C++ wrapper for persistent kernel
        cpp_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <ATen/cuda/CUDAContext.h>
        
        // Forward declare persistent kernel launcher
        void launch_fused_snn_ssm_persistent(
            const float* input,
            float* output,
            const float* ssm_A,
            const float* ssm_B,
            const float* ssm_C,
            const float* ssm_D,
            int batch_size,
            int seq_len,
            int hidden_size,
            int state_size,
            cudaStream_t stream
        );
        
        // PyTorch wrapper
        torch::Tensor fused_snn_ssm_persistent_wrapper(
            torch::Tensor input,
            torch::Tensor ssm_A,
            torch::Tensor ssm_B,
            torch::Tensor ssm_C,
            torch::Tensor ssm_D
        ) {
            // Get dimensions
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int hidden_size = input.size(2);
            int state_size = ssm_A.size(0);
            
            // Allocate output
            auto output = torch::zeros_like(input);
            
            // Get CUDA stream
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());
            
            // Launch persistent kernel (single call for entire sequence!)
            launch_fused_snn_ssm_persistent(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                ssm_A.data_ptr<float>(),
                ssm_B.data_ptr<float>(),
                ssm_C.data_ptr<float>(),
                ssm_D.data_ptr<float>(),
                batch_size,
                seq_len,
                hidden_size,
                state_size,
                stream
            );
            
            return output;
        }
        """
        
        print("🔧 Compiling persistent CUDA kernel (Phase 2.1)...")
        
        # Compile
        persistent_module = load_inline(
            name='fused_snn_ssm_persistent',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=['fused_snn_ssm_persistent_wrapper'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-arch=sm_87',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '-maxrregcount=64',  # Optimize register usage for persistent state
            ],
            extra_cflags=['-O3', '-std=c++17'],
            verbose=False,
            with_cuda=True
        )
        
        print("✅ Persistent kernel compiled successfully!")
        print("   Expected speedup: 3-5x vs baseline")
        return persistent_module
        
    except Exception as e:
        print(f"⚠️  Failed to compile persistent kernel: {e}")
        return None


class PersistentSNNSSMLayer(nn.Module):
    """
    Persistent kernel version - processes entire sequence in single GPU call
    PHASE 2.1 optimization
    """
    
    def __init__(self, hidden_size, state_size, use_cuda=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Initialize SSM parameters
        self.ssm_A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.ssm_B = nn.Parameter(torch.randn(state_size, 1) * 0.1)
        self.ssm_C = nn.Parameter(torch.randn(1, state_size) * 0.1)
        self.ssm_D = nn.Parameter(torch.randn(1, 1) * 0.01)
        
        # Try to load persistent kernel
        self.cuda_module = None
        if self.use_cuda:
            self.cuda_module = load_persistent_kernel()
            if self.cuda_module is None:
                print("⚠️  Persistent kernel unavailable")
                self.use_cuda = False
    
    def forward(self, input):
        """
        Forward pass with persistent kernel.
        
        Args:
            input: [batch_size, seq_len, hidden_size]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        if self.use_cuda and self.cuda_module is not None:
            return self.cuda_module.fused_snn_ssm_persistent_wrapper(
                input,
                self.ssm_A,
                self.ssm_B,
                self.ssm_C,
                self.ssm_D
            )
        else:
            # Fallback to standard fused kernel
            from .fused_ops import forward_pytorch
            
            batch_size, seq_len, hidden_size = input.shape
            snn_v_membrane = torch.zeros(batch_size, hidden_size, device=input.device)
            snn_refrac_timer = torch.zeros(batch_size, hidden_size, device=input.device)
            ssm_state = torch.zeros(batch_size, self.state_size, device=input.device)
            
            return forward_pytorch(
                input, snn_v_membrane, snn_refrac_timer, ssm_state,
                self.ssm_A, self.ssm_B, self.ssm_C, self.ssm_D
            )


def benchmark_persistent_vs_baseline():
    """Compare persistent kernel vs baseline"""
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "="*70)
    print("PHASE 2.1: Persistent Kernel Benchmark")
    print("="*70)
    
    device = torch.device('cuda')
    
    # Test configurations
    configs = [
        {"name": "Small", "batch": 2, "seq": 10, "hidden": 64, "state": 32},
        {"name": "Medium", "batch": 1, "seq": 64, "hidden": 512, "state": 64},
        {"name": "Large (Model)", "batch": 1, "seq": 512, "hidden": 1024, "state": 128},
    ]
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {cfg['name']}")
        print(f"  Batch={cfg['batch']}, Seq={cfg['seq']}, Hidden={cfg['hidden']}, State={cfg['state']}")
        print(f"{'='*70}")
        
        input_data = torch.randn(cfg['batch'], cfg['seq'], cfg['hidden'], device=device)
        
        # Test persistent kernel
        print("\n🚀 Testing Persistent Kernel (Phase 2.1)...")
        persistent_layer = PersistentSNNSSMLayer(cfg['hidden'], cfg['state'], use_cuda=True).to(device)
        
        if persistent_layer.use_cuda:
            # Warmup
            for _ in range(3):
                _ = persistent_layer(input_data)
            
            # Benchmark
            torch.cuda.synchronize()
            num_runs = 20
            start = time.time()
            for _ in range(num_runs):
                output_persistent = persistent_layer(input_data)
            torch.cuda.synchronize()
            persistent_time = (time.time() - start) / num_runs * 1000
            
            print(f"   Persistent kernel: {persistent_time:.3f} ms")
            print(f"   Output range: [{output_persistent.min():.4f}, {output_persistent.max():.4f}]")
            
            # Compare with baseline (from fused_ops)
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
            
            print(f"   Baseline kernel: {baseline_time:.3f} ms")
            
            # Calculate speedup
            speedup = baseline_time / persistent_time
            print(f"\n   ✅ SPEEDUP: {speedup:.2f}x")
            print(f"   Expected: 3-5x (achieved: {speedup:.2f}x)")
            
            if speedup >= 3.0:
                print(f"   🎉 BREAKTHROUGH: Phase 2.1 validated!")
            elif speedup >= 1.5:
                print(f"   ✅ Good progress, further optimization possible")
            else:
                print(f"   ⚠️  Below expectations, needs investigation")
        else:
            print("   ⚠️  Persistent kernel compilation failed")
    
    print("\n" + "="*70)
    print("PHASE 2.1 Benchmark Complete")
    print("="*70)


if __name__ == "__main__":
    benchmark_persistent_vs_baseline()
