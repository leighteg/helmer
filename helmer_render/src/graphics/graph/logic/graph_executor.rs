#![allow(dead_code)]

use hashbrown::HashMap;
use wgpu::CommandEncoder;

use crate::graphics::backend::binding_backend::RenderPassCtx;
use crate::graphics::common::renderer::RenderPassTiming;
use crate::graphics::graph::logic::frame_inputs::FrameInputHub;
use crate::graphics::graph::logic::graph_exec_ctx::RenderGraphExecCtx;
use crate::graphics::graph::logic::render_graph::{
    RenderGraph, RenderGraphCompilation, RenderGraphNode,
};

use crate::graphics::graph::logic::gpu_resource_pool::GpuResourcePool;
use web_time::Instant;

#[derive(Default, Debug, Clone, Copy)]
pub struct RenderGraphExecutionStats {
    pub pass_execute_us: u64,
    pub encoder_create_us: u64,
    pub encoder_finish_us: u64,
    pub pass_count: u32,
    pub encoder_count: u32,
    pub batch_count: u32,
}

impl RenderGraphExecutionStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

pub struct RenderGraphExecutor;

impl RenderGraphExecutor {
    fn build_batches(levels: &[Vec<usize>], encoder_batch_size: usize) -> Vec<Vec<usize>> {
        let total_nodes = levels.iter().map(Vec::len).sum::<usize>();
        if total_nodes == 0 {
            return Vec::new();
        }

        let batch_size = if encoder_batch_size == 0 {
            total_nodes.max(1)
        } else {
            encoder_batch_size.max(1)
        };
        let mut batches = Vec::with_capacity(
            total_nodes.saturating_add(batch_size.saturating_sub(1)) / batch_size,
        );
        let mut current = Vec::with_capacity(batch_size.min(total_nodes));
        for level in levels {
            for &node_id in level {
                current.push(node_id);
                if current.len() == batch_size {
                    batches.push(std::mem::take(&mut current));
                    current = Vec::with_capacity(batch_size.min(total_nodes));
                }
            }
        }
        if !current.is_empty() {
            batches.push(current);
        }
        batches
    }

    pub fn execute(
        graph: &RenderGraph,
        compilation: &RenderGraphCompilation,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &mut GpuResourcePool,
        backend: &dyn crate::graphics::backend::binding_backend::BindingBackend,
        frame_index: u32,
        frame_inputs: &FrameInputHub,
        encoder_batch_size: usize,
        pass_overrides: Option<&HashMap<String, bool>>,
        mut exec_stats: Option<&mut RenderGraphExecutionStats>,
        mut pass_timings: Option<&mut Vec<RenderPassTiming>>,
    ) -> Vec<wgpu::CommandBuffer> {
        let batches = Self::build_batches(&compilation.levels, encoder_batch_size);
        let mut command_buffers = Vec::with_capacity(batches.len());
        let pass_order = &compilation.pass_order;
        let pass_names = &compilation.pass_names;
        if let Some(entries) = pass_timings.as_mut() {
            entries.clear();
        }
        if let Some(stats) = exec_stats.as_mut() {
            stats.reset();
        }

        for (batch_idx, batch) in batches.iter().enumerate() {
            let label = format!("RenderGraph/Encoder/B{batch_idx}");
            let encoder_start = if exec_stats.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&label),
            });
            if let Some(stats) = exec_stats.as_mut() {
                if let Some(start) = encoder_start {
                    stats.encoder_create_us = stats
                        .encoder_create_us
                        .saturating_add(start.elapsed().as_micros() as u64);
                }
                stats.encoder_count = stats.encoder_count.saturating_add(1);
                stats.batch_count = stats.batch_count.saturating_add(1);
            }

            for node_id in batch {
                let node = &graph.nodes()[*node_id];
                let name = pass_names.get(*node_id).copied().unwrap_or("unknown");
                let enabled = pass_overrides
                    .and_then(|overrides| overrides.get(name).copied())
                    .unwrap_or(true);
                let measure_pass = enabled && (pass_timings.is_some() || exec_stats.is_some());
                let start_time = if measure_pass {
                    Some(Instant::now())
                } else {
                    None
                };
                if enabled {
                    Self::execute_node(
                        node,
                        &mut encoder,
                        device,
                        queue,
                        pool,
                        backend,
                        frame_index,
                        frame_inputs,
                    );
                }
                let duration_us = start_time
                    .map(|start| start.elapsed().as_micros() as u64)
                    .unwrap_or(0);
                if let Some(stats) = exec_stats.as_mut() {
                    if enabled {
                        stats.pass_execute_us = stats.pass_execute_us.saturating_add(duration_us);
                        stats.pass_count = stats.pass_count.saturating_add(1);
                    }
                }
                if let Some(entries) = pass_timings.as_mut() {
                    entries.push(RenderPassTiming {
                        name: name.to_string(),
                        order: pass_order.get(*node_id).copied().unwrap_or(0) as usize,
                        enabled,
                        duration_us,
                    });
                }
            }

            let finish_start = if exec_stats.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            command_buffers.push(encoder.finish());
            if let Some(stats) = exec_stats.as_mut() {
                if let Some(start) = finish_start {
                    stats.encoder_finish_us = stats
                        .encoder_finish_us
                        .saturating_add(start.elapsed().as_micros() as u64);
                }
            }
        }

        command_buffers
    }

    fn execute_node(
        node: &RenderGraphNode,
        encoder: &mut CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pool: &mut GpuResourcePool,
        backend: &dyn crate::graphics::backend::binding_backend::BindingBackend,
        frame_index: u32,
        frame_inputs: &FrameInputHub,
    ) {
        // Build execution context
        let rpctx = RenderPassCtx {
            backend,
            pool,
            device,
            queue,
            frame_index,
            frame_inputs,
        };

        let mut exec_ctx = RenderGraphExecCtx { rpctx, encoder };

        // Execute the pass
        node.pass.execute(&mut exec_ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::RenderGraphExecutor;

    #[test]
    fn zero_batch_size_flattens_all_levels_into_one_encoder_batch() {
        let batches = RenderGraphExecutor::build_batches(&[vec![0, 1], vec![2], vec![3, 4]], 0);
        assert_eq!(batches, vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn explicit_batch_size_chunks_across_level_boundaries() {
        let batches = RenderGraphExecutor::build_batches(&[vec![0, 1], vec![2], vec![3, 4]], 2);
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }
}
