"""
TensorNeuramorphic - Python Binding for Fused SNN-SSM Kernel (FIXED)
=====================================================================

Provides PyTorch interface to custom CUDA kernel.
FIXES: cudaStream_t declaration, PYBIND11 duplication, function naming
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import time


# ============================================================
# CUDA Kernel Loading (FIXED)
# ============================================================

def load_fused_kernel():
    """Load the fused SNN-SSM CUDA kernel with proper headers"""
    
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(kernel_dir, "kernels", "fused_snn_ssm.cu")
    
    # Check if kernel exists
    if not os.path.exists(cuda_file):
        print(f"⚠️  CUDA kernel not found at {cuda_file}")
        print("   Using PyTorch fallback implementation")
        return None
    
    try:
        # Read CUDA source
        with open(cuda_file, 'r') as f:
            cuda_source = f.read()
        
        # Fixed PyTorch wrapper code
        cpp_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <ATen/cuda/CUDAContext.h>
        
        // Forward declare the CUDA launcher
        void launch_fused_snn_ssm(
            const float* input,
            float* snn_v_membrane,
            float* snn_refrac_timer,
            float* ssm_state,
            const float* ssm_A,
            const float* ssm_B,
            const float* ssm_C,
            const float* ssm_D,
            float* output,
            int batch_size,
            int seq_len,
            int hidden_size,
            int state_size,
            cudaStream_t stream
        );
        
        // Renamed to avoid conflict with std::forward
        torch::Tensor fused_snn_ssm_forward_wrapper(
            torch::Tensor input,
            torch::Tensor snn_v_membrane,
            torch::Tensor snn_refrac_timer,
            torch::Tensor ssm_state,
            torch::Tensor ssm_A,
            torch::Tensor ssm_B,
            torch::Tensor ssm_C,
            torch::Tensor ssm_D
        ) {
            // Get dimensions
            int batch_size = input.size(0);
            int seq_len = input.size(1);
            int hidden_size = input.size(2);
            int state_size = ssm_state.size(1);
            
            // Allocate output
            auto output = torch::zeros_like(input);
            
            // Get CUDA stream with proper API for PyTorch 2.1
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());
            
            // Launch kernel
            launch_fused_snn_ssm(
                input.data_ptr<float>(),
                snn_v_membrane.data_ptr<float>(),
                snn_refrac_timer.data_ptr<float>(),
                ssm_state.data_ptr<float>(),
                ssm_A.data_ptr<float>(),
                ssm_B.data_ptr<float>(),
                ssm_C.data_ptr<float>(),
                ssm_D.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size,
                seq_len,
                hidden_size,
                state_size,
                stream
            );
            
            return output;
        }
        """
        
        print("🔧 Compiling CUDA kernel (this may take 30-60 seconds)...")
        
        # Compile with proper CUDA flags
        # NOTE: load_inline automatically generates PYBIND11_MODULE, don't include it in cpp_source
        fused_module = load_inline(
            name='fused_snn_ssm',
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            functions=['fused_snn_ssm_forward_wrapper'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-arch=sm_87',  # Orin Ampere
                '-std=c++17',
                '--expt-relaxed-constexpr',
            ],
            extra_cflags=['-O3', '-std=c++17'],
            verbose=False,
            with_cuda=True
        )
        
        print("✅ CUDA kernel compiled successfully!")
        return fused_module
        
    except Exception as e:
        print(f"⚠️  Failed to compile CUDA kernel: {e}")
        print("   Using PyTorch fallback implementation")
        return None


# ============================================================
# PyTorch Fallback Implementation (with proper initialization)
# ============================================================

def forward_pytorch(input, snn_v_membrane, snn_refrac_timer, ssm_state, 
                   ssm_A, ssm_B, ssm_C, ssm_D):
    """
    Pure PyTorch implementation of fused SNN-SSM.
    Used as fallback when CUDA compilation fails.
    """
    batch_size, seq_len, hidden_size = input.shape
    state_size = ssm_state.size(1)
    
    # Allocate output
    output = torch.zeros_like(input)
    
    # SNN parameters (LIF neuron)
    tau_membrane = 20.0
    tau_synapse = 5.0
    v_threshold = 1.0
    v_reset = 0.0
    refrac_period = 2.0
    
    # Temporal loop
    for t in range(seq_len):
        # Get input for this timestep
        x_t = input[:, t, :]  # [batch, hidden]
        
        # === SNN Dynamics (LIF) ===
        # Update membrane potential (leaky integration)
        leak = torch.exp(torch.tensor(-1.0 / tau_membrane))
        snn_v_membrane = leak * snn_v_membrane + (1 - leak) * x_t
        
        # Check for spikes (threshold crossing)
        spike_mask = (snn_v_membrane >= v_threshold) & (snn_refrac_timer <= 0)
        spikes = spike_mask.float()
        
        # Reset spiked neurons
        snn_v_membrane = torch.where(spike_mask, 
                                     torch.full_like(snn_v_membrane, v_reset),
                                     snn_v_membrane)
        snn_refrac_timer = torch.where(spike_mask,
                                       torch.full_like(snn_refrac_timer, refrac_period),
                                       snn_refrac_timer - 1)
        
        # === SSM Dynamics ===
        # State update: x_t+1 = A * x_t + B * u_t
        # Use spikes as input to SSM
        ssm_input = spikes.unsqueeze(2)  # [batch, hidden, 1]
        
        # Expand ssm_state to match hidden_size
        # Each hidden unit has its own state vector
        ssm_state_expanded = ssm_state.unsqueeze(1).expand(batch_size, hidden_size, state_size)
        
        # Apply SSM update
        # A: [state_size, state_size], B: [state_size, 1]
        new_state = torch.matmul(ssm_state_expanded, ssm_A.t()) + \
                   torch.matmul(ssm_input, ssm_B.t())
        
        # Output: y_t = C * x_t + D * u_t
        # C: [1, state_size], D: [1, 1]
        ssm_output = torch.matmul(new_state, ssm_C.t()).squeeze(-1) + \
                    torch.matmul(ssm_input, ssm_D.t()).squeeze(-1)
        
        # Update state (average across hidden dimension)
        ssm_state = new_state.mean(dim=1)
        
        # Store output
        output[:, t, :] = ssm_output
    
    return output


# ============================================================
# Main Layer Class
# ============================================================

class FusedSNNSSMLayer(nn.Module):
    """
    PyTorch layer wrapping the fused SNN-SSM CUDA kernel.
    Falls back to PyTorch if CUDA is unavailable.
    """
    
    def __init__(self, hidden_size, state_size, use_cuda=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Initialize SSM parameters (proper scale)
        self.ssm_A = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.ssm_B = nn.Parameter(torch.randn(state_size, 1) * 0.1)
        self.ssm_C = nn.Parameter(torch.randn(1, state_size) * 0.1)
        self.ssm_D = nn.Parameter(torch.randn(1, 1) * 0.01)
        
        # Try to load CUDA kernel
        self.cuda_module = None
        if self.use_cuda:
            self.cuda_module = load_fused_kernel()
            if self.cuda_module is None:
                print("⚠️  CUDA kernel unavailable, using PyTorch fallback")
                self.use_cuda = False
    
    def forward(self, input):
        """
        Forward pass through fused SNN-SSM layer.
        
        Args:
            input: [batch_size, seq_len, hidden_size]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = input.shape
        
        # Initialize states
        snn_v_membrane = torch.zeros(batch_size, hidden_size, device=input.device)
        snn_refrac_timer = torch.zeros(batch_size, hidden_size, device=input.device)
        ssm_state = torch.zeros(batch_size, self.state_size, device=input.device)
        
        # Use CUDA kernel if available
        if self.use_cuda and self.cuda_module is not None:
            return self.cuda_module.fused_snn_ssm_forward_wrapper(
                input, snn_v_membrane, snn_refrac_timer, ssm_state,
                self.ssm_A, self.ssm_B, self.ssm_C, self.ssm_D
            )
        else:
            # PyTorch fallback
            return forward_pytorch(
                input, snn_v_membrane, snn_refrac_timer, ssm_state,
                self.ssm_A, self.ssm_B, self.ssm_C, self.ssm_D
            )


# ============================================================
# Benchmark Function
# ============================================================

def benchmark_fused_kernel(batch_size=2, seq_len=10, hidden_size=64, 
                           state_size=32, num_warmup=5, num_runs=20):
    """
    Benchmark the fused kernel against PyTorch implementation.
    
    Returns:
        dict with timing results and speedup
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test input
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Initialize states
    snn_v_membrane = torch.zeros(batch_size, hidden_size, device=device)
    snn_refrac_timer = torch.zeros(batch_size, hidden_size, device=device)
    ssm_state = torch.zeros(batch_size, state_size, device=device)
    
    # Initialize SSM parameters (proper scale)
    ssm_A = torch.randn(state_size, state_size, device=device) * 0.01
    ssm_B = torch.randn(state_size, 1, device=device) * 0.1
    ssm_C = torch.randn(1, state_size, device=device) * 0.1
    ssm_D = torch.randn(1, 1, device=device) * 0.01
    
    # Benchmark PyTorch implementation
    print("\n📊 Benchmarking PyTorch implementation...")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(num_warmup):
        _ = forward_pytorch(input_data, snn_v_membrane.clone(), 
                           snn_refrac_timer.clone(), ssm_state.clone(),
                           ssm_A, ssm_B, ssm_C, ssm_D)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        output_pytorch = forward_pytorch(input_data, snn_v_membrane.clone(), 
                                        snn_refrac_timer.clone(), ssm_state.clone(),
                                        ssm_A, ssm_B, ssm_C, ssm_D)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_runs * 1000  # ms
    
    print(f"   PyTorch: {pytorch_time:.2f} ms/iteration")
    print(f"   Output shape: {output_pytorch.shape}")
    print(f"   Output range: [{output_pytorch.min():.4f}, {output_pytorch.max():.4f}]")
    
    # Try CUDA kernel
    cuda_time = None
    speedup = None
    cuda_module = load_fused_kernel()
    
    if cuda_module is not None:
        print("\n📊 Benchmarking CUDA kernel...")
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(num_warmup):
            _ = cuda_module.fused_snn_ssm_forward_wrapper(input_data, snn_v_membrane.clone(), 
                                   snn_refrac_timer.clone(), ssm_state.clone(),
                                   ssm_A, ssm_B, ssm_C, ssm_D)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            output_cuda = cuda_module.fused_snn_ssm_forward_wrapper(input_data, snn_v_membrane.clone(), 
                                             snn_refrac_timer.clone(), ssm_state.clone(),
                                             ssm_A, ssm_B, ssm_C, ssm_D)
        torch.cuda.synchronize()
        cuda_time = (time.time() - start) / num_runs * 1000  # ms
        
        speedup = pytorch_time / cuda_time
        
        print(f"   CUDA: {cuda_time:.2f} ms/iteration")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Output shape: {output_cuda.shape}")
        print(f"   Output range: [{output_cuda.min():.4f}, {output_cuda.max():.4f}]")
    
    return {
        'pytorch_ms': pytorch_time,
        'cuda_ms': cuda_time,
        'speedup': speedup,
        'input_shape': input_data.shape,
        'output_shape': output_pytorch.shape
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TensorNeuramorphic - Fused SNN-SSM Kernel (FIXED)")
    print("="*60)
    
    results = benchmark_fused_kernel(
        batch_size=2,
        seq_len=10,
        hidden_size=64,
        state_size=32,
        num_runs=20
    )
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  PyTorch: {results['pytorch_ms']:.2f} ms")
    if results['cuda_ms'] is not None:
        print(f"  CUDA: {results['cuda_ms']:.2f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
    print("="*60)
