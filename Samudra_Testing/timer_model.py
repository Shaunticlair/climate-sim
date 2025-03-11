import torch
import torch.nn as nn
import numpy as np
import time
from collections import defaultdict
from itertools import tee

# Import the original model classes 
from model import pairwise, CappedGELU, BilinearUpsample, AvgPool, BaseSamudra, ConvNeXtBlock
from model import generate_model_rollout  # Keep the original rollout function

# Global timing variables
parallel_time = 0.0
sequential_time = 0.0
timing_stats = defaultdict(lambda: {"parallel": 0.0, "sequential": 0.0})

class TimedConvNeXtBlock(ConvNeXtBlock):
    """Extension of ConvNeXtBlock with timing instrumentation"""
    def forward(self, x):
        global parallel_time, sequential_time, timing_stats
        
        # Skip connection (sequential - channel dependency)
        t_start = time.time()
        skip = self.skip_module(x)
        t_end = time.time()
        sequential_time += t_end - t_start
        timing_stats["ConvNeXtBlock_skip"]["sequential"] += t_end - t_start
        
        # 1st convolution (sequential - channel mixing)
        t_start = time.time()
        first_conv = self.convblock[0]
        if first_conv.kernel_size[0] != 1:
            x = torch.nn.functional.pad(
                x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
            )
            x = torch.nn.functional.pad(
                x, (0, 0, self.N_pad, self.N_pad), mode="constant"
            )
        x = first_conv(x)
        x = self.convblock[1](x)  # BatchNorm
        x = self.convblock[2](x)  # Activation
        t_end = time.time()
        sequential_time += t_end - t_start
        timing_stats["ConvNeXtBlock_first_conv"]["sequential"] += t_end - t_start
        
        # Middle part (parallelizable across channels)
        t_start = time.time()
        middle_conv = self.convblock[3]
        if middle_conv.kernel_size[0] != 1:
            x = torch.nn.functional.pad(
                x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
            )
            x = torch.nn.functional.pad(
                x, (0, 0, self.N_pad, self.N_pad), mode="constant"
            )
        x = middle_conv(x)
        x = self.convblock[4](x)  # BatchNorm
        x = self.convblock[5](x)  # Activation
        t_end = time.time()
        parallel_time += t_end - t_start
        timing_stats["ConvNeXtBlock_middle_conv"]["parallel"] += t_end - t_start
        
        # Final 1x1 conv (sequential - channel dependency)
        t_start = time.time()
        x = self.convblock[6](x)  # 1x1 conv
        result = skip + x  # Residual connection
        t_end = time.time()
        sequential_time += t_end - t_start
        timing_stats["ConvNeXtBlock_final_conv"]["sequential"] += t_end - t_start
        
        return result


class TimedSamudra(BaseSamudra):
    """Extension of Samudra with timing instrumentation"""
    def __init__(
        self,
        wet,
        hist,
        core_block=TimedConvNeXtBlock,  # Use the timed version
        down_sampling_block=AvgPool,
        up_sampling_block=BilinearUpsample,
        activation=CappedGELU,
        ch_width=[157,200,250,300,400],
        n_out=77,
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    ):
        super(BaseSamudra, self).__init__()
        
        # Initialize from BaseSamudra (skipping the original Samudra init)
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.ch_width = ch_width
        self.wet = wet
        self.N_pad = int((last_kernel_size - 1) / 2)
        self.pad = pad
        self.pred_residuals = pred_residuals
        self.hist = hist
        self.input_channels = ch_width[0]
        self.output_channels = n_out
        
        # Build the layers just like in Samudra
        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(down_sampling_block())
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(up_sampling_block(in_channels=b, out_channels=b))
        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(
                core_block(
                    a,
                    b,
                    dilation=dilation[i],
                    n_layers=n_layers[i],
                    activation=activation,
                    pad=pad,
                )
            )
            layers.append(up_sampling_block(in_channels=b, out_channels=b))
        layers.append(
            core_block(
                b,
                b,
                dilation=dilation[i],
                n_layers=n_layers[i],
                activation=activation,
                pad=pad,
            )
        )
        layers.append(torch.nn.Conv2d(b, n_out, last_kernel_size))

        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width) - 1)

    def forward_once(self, fts):
        global parallel_time, sequential_time, timing_stats
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        
        for l in self.layers:
            crop = fts.shape[2:]
            
            # ConvNeXtBlock already has timing instrumentation inside
            if isinstance(l, TimedConvNeXtBlock):
                fts = l(fts)
                if count < self.num_steps:
                    temp[count] = fts
                    count += 1
            
            # Padding operations (sequential)
            elif isinstance(l, nn.Conv2d):
                t_start = time.time()
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
                fts = l(fts)
                t_end = time.time()
                sequential_time += t_end - t_start
                timing_stats["Conv2d"]["sequential"] += t_end - t_start
            
            # Pooling operations (channel-independent - parallelizable)
            elif isinstance(l, AvgPool):
                t_start = time.time()
                fts = l(fts)
                t_end = time.time()
                parallel_time += t_end - t_start
                timing_stats["AvgPool"]["parallel"] += t_end - t_start
            
            # Upsampling operations with skip connections (mixed)
            elif isinstance(l, BilinearUpsample):
                # Upsampling itself is parallelizable
                t_start = time.time()
                fts = l(fts)
                t_end = time.time()
                parallel_time += t_end - t_start
                timing_stats["BilinearUpsample_op"]["parallel"] += t_end - t_start
                
                # Skip connection handling is sequential
                t_start = time.time()
                crop = np.array(fts.shape[2:])
                shape = np.array(
                    temp[int(2 * self.num_steps - count - 1)].shape[2:]
                )
                pads = shape - crop
                pads = [
                    pads[1] // 2,
                    pads[1] - pads[1] // 2,
                    pads[0] // 2,
                    pads[0] - pads[0] // 2,
                ]
                fts = nn.functional.pad(fts, pads)
                fts += temp[int(2 * self.num_steps - count - 1)]  # Skip connection
                count += 1
                t_end = time.time()
                sequential_time += t_end - t_start
                timing_stats["BilinearUpsample_skip"]["sequential"] += t_end - t_start
        
        # Final weighting (parallelizable)
        t_start = time.time()
        result = torch.mul(fts, self.wet)
        t_end = time.time()
        parallel_time += t_end - t_start
        timing_stats["final_mul"]["parallel"] += t_end - t_start
        
        return result

    def forward(self, inputs, output_only_last=False, loss_fn=None):
        global parallel_time, sequential_time, timing_stats
        outputs = []
        loss = None
        N, C, H, W = inputs[0].shape

        for step in range(len(inputs) // 2):
            # Input preparation (sequential)
            t_start = time.time()
            if step == 0:
                input_tensor = inputs[0]
            else:
                inputs_0 = outputs[-1]
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[2 * step][:, self.output_channels:],
                    ],
                    dim=1,
                )
            t_end = time.time()
            sequential_time += t_end - t_start
            timing_stats["input_preparation"]["sequential"] += t_end - t_start

            # Forward pass - timed internally in forward_once
            decodings = self.forward_once(input_tensor)
            
            # Output processing (sequential due to residual connections)
            t_start = time.time()
            if self.pred_residuals:
                reshaped = input_tensor[:, :self.output_channels] + decodings
            else:
                reshaped = decodings
                
            if loss_fn is not None:
                assert (
                    reshaped.shape == inputs[2 * step + 1].shape
                ), f"Output shape is {reshaped.shape} but should be {inputs[2 * step + 1].shape}"
                if loss is None:
                    loss = loss_fn(reshaped, inputs[2 * step + 1])
                else:
                    loss += loss_fn(reshaped, inputs[2 * step + 1])
            
            outputs.append(reshaped)
            t_end = time.time()
            sequential_time += t_end - t_start
            timing_stats["output_processing"]["sequential"] += t_end - t_start

        if loss_fn is None:
            if output_only_last:
                res = outputs[-1]
            else:
                res = outputs
            return res
        else:
            return loss
    
    def inference(self, inputs, initial_input=None, num_steps=None, output_only_last=False, device="cpu"):
        global parallel_time, sequential_time, timing_stats
        outputs = []
        
        for step in range(num_steps):
            # Input preparation (sequential)
            t_start = time.time()
            if step == 0:
                input_tensor = inputs[0][0].to(device=device)
                if initial_input is not None:
                    input_tensor[:, :self.output_channels] = initial_input
            else:
                inputs_0 = outputs[-1].unsqueeze(0)
                input_tensor = torch.cat(
                    [
                        inputs_0,
                        inputs[step][0][:, self.output_channels:].to(device=device),
                    ],
                    dim=1,
                )
            t_end = time.time()
            sequential_time += t_end - t_start
            timing_stats["inference_input_prep"]["sequential"] += t_end - t_start

            # Forward pass - timed internally
            decodings = self.forward_once(input_tensor)
            
            # Output processing (sequential)
            t_start = time.time()
            if self.pred_residuals:
                reshaped = input_tensor[0, :self.output_channels].to(device=device) + decodings.squeeze(0)
            else:
                reshaped = decodings.squeeze(0)
            
            outputs.append(reshaped)
            t_end = time.time()
            sequential_time += t_end - t_start
            timing_stats["inference_output_proc"]["sequential"] += t_end - t_start

        if output_only_last:
            res = outputs[-1]
        else:
            res = outputs

        return res

# Function to reset and get timing stats
def reset_timers():
    global parallel_time, sequential_time, timing_stats
    parallel_time = 0.0
    sequential_time = 0.0
    timing_stats = defaultdict(lambda: {"parallel": 0.0, "sequential": 0.0})

def get_timing_stats():
    global parallel_time, sequential_time, timing_stats
    
    # Calculate total time and parallelizable percentage
    total_time = parallel_time + sequential_time
    parallel_percentage = (parallel_time / total_time * 100) if total_time > 0 else 0
    
    # Prepare detailed stats
    detailed_stats = {k: v for k, v in timing_stats.items()}
    
    # Return a comprehensive report
    return {
        "total_time": total_time,
        "parallel_time": parallel_time,
        "sequential_time": sequential_time,
        "parallel_percentage": parallel_percentage,
        "detailed_stats": detailed_stats
    }

# Wrapped rollout function with timing
def timed_generate_model_rollout(*args, **kwargs):
    reset_timers()
    result, _ = generate_model_rollout(*args, **kwargs)
    timing_report = get_timing_stats()
    return result, timing_report
