#![allow(dead_code)]

use dyn_clone::DynClone;

use crate::graphics::backend::binding_backend::PassBindingPolicy;
use crate::graphics::graph::definition::resource_id::ResourceId;
use crate::graphics::graph::logic::graph_context::RenderGraphContext;
use crate::graphics::graph::logic::graph_exec_ctx::RenderGraphExecCtx;

/// A user-defined modular render pass.
pub trait RenderPass: DynClone + Send + Sync + 'static {
    /// Stable name for debugging / tracking.
    fn name(&self) -> &'static str;

    /// Declare dependencies: what resources you read/write.
    fn setup(&self, ctx: &mut RenderGraphContext);

    /// Run the pass (record commands).
    fn execute(&self, ctx: &mut RenderGraphExecCtx);

    /// Basic pass binding policy.
    fn binding_policy(&self) -> PassBindingPolicy {
        PassBindingPolicy::Full
    }

    /// Optional hook for passes that cache render bundles.
    fn clear_cached_bundles(&self) {}

    /// Optional hook for passes that cache render bundles keyed by resources.
    fn invalidate_cached_bundles_for_resources(&self, _resources: &[ResourceId]) {}
}
