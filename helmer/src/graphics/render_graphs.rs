use winit::dpi::PhysicalSize;

use crate::graphics::{
    config::{
        DEBUG_FLAG_DIRECT, DEBUG_FLAG_EMISSION, DEBUG_FLAG_INDIRECT, DEBUG_FLAG_REFLECTIONS,
        DEBUG_FLAG_SKY, DEBUG_FLAG_UNLIT, RenderConfig,
    },
    graph::{
        definition::{
            resource_desc::ResourceDesc, resource_flags::ResourceFlags, resource_id::ResourceId,
        },
        logic::{
            gpu_resource_pool::GpuResourcePool, pass_registry::PassGraphBuilder,
            pass_registry::PassResourceOutput,
        },
    },
    passes::{
        composite::{CompositeInputs, CompositePass},
        debug_composite::{DebugCompositeInputs, DebugCompositePass},
        depth_copy::{DepthCopyOutputs, DepthCopyPass},
        downsample::{DownsampleOutputs, DownsamplePass},
        egui::{EguiOutputs, EguiPass},
        gbuffer::{GBufferFormats, GBufferOutputs, GBufferPass},
        hiz::{HiZOutputs, HiZPass},
        lighting::{LightingOutputs, LightingPass},
        raytracing::{RayTracingOutputs, RayTracingPass},
        raytracing_composite::{RayTracingCompositeInputs, RayTracingCompositePass},
        shadow::{ShadowOutputs, ShadowPass},
        sky::{SkyOutputs, SkyPass},
        ssgi::{SsgiOutputs, SsgiPass},
        ssgi_denoise::{SsgiDenoiseOutputs, SsgiDenoisePass},
        ssgi_upsample::{SsgiUpsampleOutputs, SsgiUpsamplePass},
        ssr::{SsrOutputs, SsrPass},
    },
    renderer_common::graph::{RenderGraphBuildOutput, RenderGraphBuildParams, RenderGraphSpec},
};

#[derive(Clone, Copy, Debug)]
pub enum RenderPassToggleFlag {
    GBuffer,
    Shadow,
    DirectLighting,
    Sky,
    Ssgi,
    SsgiDenoise,
    Ssr,
    Egui,
    Occlusion,
}

impl RenderPassToggleFlag {
    pub fn get(self, config: &RenderConfig) -> bool {
        match self {
            RenderPassToggleFlag::GBuffer => config.gbuffer_pass,
            RenderPassToggleFlag::Shadow => config.shadow_pass,
            RenderPassToggleFlag::DirectLighting => config.direct_lighting_pass,
            RenderPassToggleFlag::Sky => config.sky_pass,
            RenderPassToggleFlag::Ssgi => config.ssgi_pass,
            RenderPassToggleFlag::SsgiDenoise => config.ssgi_denoise_pass,
            RenderPassToggleFlag::Ssr => config.ssr_pass,
            RenderPassToggleFlag::Egui => config.egui_pass,
            RenderPassToggleFlag::Occlusion => config.occlusion_culling,
        }
    }

    pub fn set(self, config: &mut RenderConfig, value: bool) {
        match self {
            RenderPassToggleFlag::GBuffer => config.gbuffer_pass = value,
            RenderPassToggleFlag::Shadow => config.shadow_pass = value,
            RenderPassToggleFlag::DirectLighting => config.direct_lighting_pass = value,
            RenderPassToggleFlag::Sky => config.sky_pass = value,
            RenderPassToggleFlag::Ssgi => config.ssgi_pass = value,
            RenderPassToggleFlag::SsgiDenoise => config.ssgi_denoise_pass = value,
            RenderPassToggleFlag::Ssr => config.ssr_pass = value,
            RenderPassToggleFlag::Egui => config.egui_pass = value,
            RenderPassToggleFlag::Occlusion => config.occlusion_culling = value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderPassToggle {
    pub label: &'static str,
    pub toggle: RenderPassToggleFlag,
}

#[derive(Clone, Copy, Debug)]
pub struct DebugFlagToggle {
    pub label: &'static str,
    pub mask: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderGraphTemplate {
    pub name: &'static str,
    pub label: &'static str,
    pub build: fn() -> RenderGraphSpec,
    pub pass_toggles: &'static [RenderPassToggle],
    pub debug_flags: &'static [DebugFlagToggle],
}

const DEFAULT_GRAPH_PASSES: &[RenderPassToggle] = &[
    RenderPassToggle {
        label: "GBuffer",
        toggle: RenderPassToggleFlag::GBuffer,
    },
    RenderPassToggle {
        label: "Shadow",
        toggle: RenderPassToggleFlag::Shadow,
    },
    RenderPassToggle {
        label: "Direct Lighting",
        toggle: RenderPassToggleFlag::DirectLighting,
    },
    RenderPassToggle {
        label: "Sky",
        toggle: RenderPassToggleFlag::Sky,
    },
    RenderPassToggle {
        label: "SSGI",
        toggle: RenderPassToggleFlag::Ssgi,
    },
    RenderPassToggle {
        label: "SSGI Denoise",
        toggle: RenderPassToggleFlag::SsgiDenoise,
    },
    RenderPassToggle {
        label: "SSR",
        toggle: RenderPassToggleFlag::Ssr,
    },
    RenderPassToggle {
        label: "Egui",
        toggle: RenderPassToggleFlag::Egui,
    },
    RenderPassToggle {
        label: "Occlusion (Hi-Z)",
        toggle: RenderPassToggleFlag::Occlusion,
    },
];

const DEBUG_GRAPH_FLAGS: &[DebugFlagToggle] = &[
    DebugFlagToggle {
        label: "unlit",
        mask: DEBUG_FLAG_UNLIT,
    },
    DebugFlagToggle {
        label: "direct lighting",
        mask: DEBUG_FLAG_DIRECT,
    },
    DebugFlagToggle {
        label: "indirect diffuse",
        mask: DEBUG_FLAG_INDIRECT,
    },
    DebugFlagToggle {
        label: "reflections",
        mask: DEBUG_FLAG_REFLECTIONS,
    },
    DebugFlagToggle {
        label: "emission",
        mask: DEBUG_FLAG_EMISSION,
    },
    DebugFlagToggle {
        label: "sky",
        mask: DEBUG_FLAG_SKY,
    },
];

const TRACED_GRAPH_PASSES: &[RenderPassToggle] = &[RenderPassToggle {
    label: "Egui",
    toggle: RenderPassToggleFlag::Egui,
}];

const GRAPH_TEMPLATES: &[RenderGraphTemplate] = &[
    RenderGraphTemplate {
        name: "default-graph",
        label: "Default Graph",
        build: default_graph_spec,
        pass_toggles: DEFAULT_GRAPH_PASSES,
        debug_flags: &[],
    },
    RenderGraphTemplate {
        name: "debug-graph",
        label: "Debug Graph",
        build: debug_graph_spec,
        pass_toggles: DEFAULT_GRAPH_PASSES,
        debug_flags: DEBUG_GRAPH_FLAGS,
    },
    RenderGraphTemplate {
        name: "traced-graph",
        label: "Traced Graph",
        build: traced_graph_spec,
        pass_toggles: TRACED_GRAPH_PASSES,
        debug_flags: &[],
    },
];

pub fn graph_templates() -> &'static [RenderGraphTemplate] {
    GRAPH_TEMPLATES
}

pub fn template_for_graph(name: &str) -> Option<&'static RenderGraphTemplate> {
    GRAPH_TEMPLATES
        .iter()
        .find(|template| template.name == name)
}

impl PassResourceOutput for GBufferOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![
            self.normal,
            self.albedo,
            self.mra,
            self.emission,
            self.depth,
            self.depth_copy,
        ]
    }
}

impl PassResourceOutput for DepthCopyOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.depth_copy]
    }
}

impl PassResourceOutput for ShadowOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.map, self.depth]
    }
}

impl PassResourceOutput for SkyOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.sky]
    }
}

impl PassResourceOutput for LightingOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.lighting, self.lighting_diffuse]
    }
}

impl PassResourceOutput for HiZOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.hiz]
    }
}

impl PassResourceOutput for DownsampleOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.depth, self.normal, self.albedo, self.lighting_diffuse]
    }
}

impl PassResourceOutput for SsgiOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.raw_half]
    }
}

impl PassResourceOutput for SsgiDenoiseOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.denoised_half]
    }
}

impl PassResourceOutput for SsgiUpsampleOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.upsampled]
    }
}

impl PassResourceOutput for SsrOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.reflection]
    }
}

impl PassResourceOutput for RayTracingOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.accumulation]
    }
}

impl PassResourceOutput for CompositeInputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        let mut ids = vec![
            self.direct_lighting,
            self.albedo,
            self.emission,
            self.normal,
            self.mra,
            self.depth,
            self.sky,
            self.swapchain,
        ];
        if let Some(id) = self.ssgi {
            ids.push(id);
        }
        if let Some(id) = self.ssr {
            ids.push(id);
        }
        ids
    }
}

impl PassResourceOutput for DebugCompositeInputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        let mut ids = vec![
            self.direct_lighting,
            self.albedo,
            self.emission,
            self.normal,
            self.mra,
            self.depth,
            self.sky,
            self.swapchain,
        ];
        if let Some(id) = self.ssgi {
            ids.push(id);
        }
        if let Some(id) = self.ssr {
            ids.push(id);
        }
        ids
    }
}

impl PassResourceOutput for RayTracingCompositeInputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.accumulation, self.swapchain]
    }
}

impl PassResourceOutput for EguiOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.swapchain]
    }
}

/// Default graph used by the renderer: shadow -> gbuffer -> lighting -> post.
pub fn default_graph_spec() -> RenderGraphSpec {
    RenderGraphSpec::unique("default-graph", |params, pool| {
        build_default_graph(params, pool)
    })
}

/// Debug graph: same as default but composites with debug toggles.
pub fn debug_graph_spec() -> RenderGraphSpec {
    RenderGraphSpec::unique("debug-graph", |params, pool| {
        build_debug_graph(params, pool)
    })
}

/// Traced graph: BVH-traced path with accumulation.
pub fn traced_graph_spec() -> RenderGraphSpec {
    RenderGraphSpec::unique("traced-graph", |params, pool| {
        build_traced_graph(params, pool)
    })
}

fn build_default_graph(
    params: &RenderGraphBuildParams,
    pool: &mut GpuResourcePool,
) -> RenderGraphBuildOutput {
    let size: PhysicalSize<u32> = params.surface_size;
    let toggles = params.config;
    let mut ssgi_history = None;
    let mut hiz_id = None;

    if toggles.ssgi_pass {
        let (ssgi_history_desc, mut ssgi_history_hints) = ResourceDesc::Texture2D {
            width: (size.width / 2).max(1),
            height: (size.height / 2).max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
        .with_hints();
        ssgi_history_hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let history_id = pool.create_logical(ssgi_history_desc, Some(ssgi_history_hints), 0, None);
        ssgi_history = Some(history_id);
    }

    let gbuffer_formats = GBufferFormats::select(
        params
            .device_caps
            .limits
            .max_color_attachment_bytes_per_sample,
    );
    let mut builder = PassGraphBuilder::new(pool);

    builder.add::<ShadowPass, ShadowOutputs, _>(|pool, _| {
        let pass = ShadowPass::new(
            pool,
            params.shadow_format,
            params.shadow_map_resolution,
            params.shadow_cascade_count,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    builder.add::<GBufferPass, GBufferOutputs, _>(|pool, _| {
        let pass = GBufferPass::new(
            pool,
            size.width,
            size.height,
            gbuffer_formats,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if !gbuffer_formats.output_depth_copy {
        builder.add::<DepthCopyPass, DepthCopyOutputs, _>(|_, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass =
                DepthCopyPass::new(gbuffer.depth, gbuffer.depth_copy, size.width, size.height);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    builder.add::<SkyPass, SkyOutputs, _>(|pool, store| {
        let gbuffer = store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let pass = SkyPass::new(
            pool,
            gbuffer.depth_copy,
            size.width,
            size.height,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if toggles.occlusion_culling {
        let hiz = builder.add::<HiZPass, HiZOutputs, _>(|pool, store| {
            let gbuffer = store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = HiZPass::new(pool, gbuffer.depth_copy, size.width, size.height);
            let outputs = pass.outputs();
            (pass, outputs)
        });
        hiz_id = Some(hiz.outputs.hiz);
    }

    builder.add::<LightingPass, LightingOutputs, _>(|pool, store| {
        let gbuffer = *store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let shadow = *store
            .outputs::<ShadowOutputs>()
            .expect("Shadow pass missing");
        let pass = LightingPass::new(
            pool,
            gbuffer,
            shadow,
            size.width,
            size.height,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    let mut ssgi_final: Option<ResourceId> = None;
    if toggles.ssgi_pass {
        let history_id = ssgi_history.expect("ssgi history id missing");

        builder.add::<DownsamplePass, DownsampleOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let lighting = store
                .outputs::<LightingOutputs>()
                .expect("Lighting pass missing");
            let pass = DownsamplePass::new(
                pool,
                gbuffer,
                lighting.lighting_diffuse,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });

        builder.add::<SsgiPass, SsgiOutputs, _>(|pool, store| {
            let downsample = *store
                .outputs::<DownsampleOutputs>()
                .expect("Downsample pass missing");
            let pass = SsgiPass::new(
                pool,
                downsample,
                history_id,
                size.width,
                size.height,
                params.blue_noise_view.clone(),
                params.blue_noise_sampler.clone(),
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });

        let denoised_input = if toggles.ssgi_denoise_pass {
            let denoised = builder.add::<SsgiDenoisePass, SsgiDenoiseOutputs, _>(|pool, store| {
                let ssgi_outputs = store.outputs::<SsgiOutputs>().expect("SSGI pass missing");
                let downsample = store
                    .outputs::<DownsampleOutputs>()
                    .expect("Downsample pass missing");
                let pass = SsgiDenoisePass::new(
                    pool,
                    ssgi_outputs.raw_half,
                    downsample.depth,
                    downsample.normal,
                    history_id,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            });
            Some(denoised.outputs.denoised_half)
        } else {
            None
        };

        let upsample =
            builder.add::<SsgiUpsamplePass, SsgiUpsampleOutputs, _>(move |pool, store| {
                let gbuffer = store
                    .outputs::<GBufferOutputs>()
                    .expect("G-buffer pass missing");
                let source = denoised_input.unwrap_or_else(|| {
                    store
                        .outputs::<SsgiOutputs>()
                        .expect("SSGI pass missing")
                        .raw_half
                });
                let pass = SsgiUpsamplePass::new(
                    pool,
                    source,
                    gbuffer.depth_copy,
                    gbuffer.normal,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            });
        ssgi_final = Some(upsample.outputs.upsampled);
    }

    let mut ssr_output = None;
    if toggles.ssr_pass {
        let ssr = builder.add::<SsrPass, SsrOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let lighting = store
                .outputs::<LightingOutputs>()
                .expect("Lighting pass missing");
            let pass = SsrPass::new(
                pool,
                gbuffer,
                lighting.lighting,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
        ssr_output = Some(ssr.outputs.reflection);
    }

    let ssgi_tex = ssgi_final;
    let ssr_tex = ssr_output;

    let composite = builder.add::<CompositePass, CompositeInputs, _>(move |pool, store| {
        let lighting = *store
            .outputs::<LightingOutputs>()
            .expect("Lighting pass missing");
        let ssgi = ssgi_tex.or_else(|| store.outputs::<SsgiUpsampleOutputs>().map(|o| o.upsampled));
        let ssr = ssr_tex.or_else(|| store.outputs::<SsrOutputs>().map(|o| o.reflection));
        let gbuffer = *store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let sky = *store.outputs::<SkyOutputs>().expect("Sky pass missing");
        let pass = CompositePass::new(
            pool,
            lighting.lighting,
            ssgi,
            ssr,
            gbuffer,
            sky.sky,
            params.surface_format,
        );
        let outputs = pass.inputs();
        (pass, outputs)
    });
    let swapchain_id = composite.outputs.swapchain;

    if toggles.egui_pass {
        builder.add::<EguiPass, EguiOutputs, _>(|pool, _store| {
            let pass = EguiPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    let (graph, passes) = builder.into_parts();
    let mut resource_ids: Vec<ResourceId> = passes.resource_ids().collect();
    if let Some(history) = ssgi_history {
        resource_ids.push(history);
    }
    resource_ids.sort_by_key(|id| id.raw());
    resource_ids.dedup_by_key(|id| id.raw());

    RenderGraphBuildOutput {
        graph,
        swapchain_id,
        resource_ids,
        hiz_id,
    }
}

fn build_debug_graph(
    params: &RenderGraphBuildParams,
    pool: &mut GpuResourcePool,
) -> RenderGraphBuildOutput {
    let size: PhysicalSize<u32> = params.surface_size;
    let toggles = params.config;
    let mut ssgi_history = None;
    let mut hiz_id = None;

    if toggles.ssgi_pass {
        let (ssgi_history_desc, mut ssgi_history_hints) = ResourceDesc::Texture2D {
            width: (size.width / 2).max(1),
            height: (size.height / 2).max(1),
            mip_levels: 1,
            layers: 1,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
        .with_hints();
        ssgi_history_hints.flags |= ResourceFlags::PREFER_RESIDENT;
        let history_id = pool.create_logical(ssgi_history_desc, Some(ssgi_history_hints), 0, None);
        ssgi_history = Some(history_id);
    }

    let gbuffer_formats = GBufferFormats::select(
        params
            .device_caps
            .limits
            .max_color_attachment_bytes_per_sample,
    );
    let mut builder = PassGraphBuilder::new(pool);

    builder.add::<ShadowPass, ShadowOutputs, _>(|pool, _| {
        let pass = ShadowPass::new(
            pool,
            params.shadow_format,
            params.shadow_map_resolution,
            params.shadow_cascade_count,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    builder.add::<GBufferPass, GBufferOutputs, _>(|pool, _| {
        let pass = GBufferPass::new(
            pool,
            size.width,
            size.height,
            gbuffer_formats,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if !gbuffer_formats.output_depth_copy {
        builder.add::<DepthCopyPass, DepthCopyOutputs, _>(|_, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass =
                DepthCopyPass::new(gbuffer.depth, gbuffer.depth_copy, size.width, size.height);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    builder.add::<SkyPass, SkyOutputs, _>(|pool, store| {
        let gbuffer = store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let pass = SkyPass::new(
            pool,
            gbuffer.depth_copy,
            size.width,
            size.height,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if toggles.occlusion_culling {
        let hiz = builder.add::<HiZPass, HiZOutputs, _>(|pool, store| {
            let gbuffer = store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = HiZPass::new(pool, gbuffer.depth_copy, size.width, size.height);
            let outputs = pass.outputs();
            (pass, outputs)
        });
        hiz_id = Some(hiz.outputs.hiz);
    }

    builder.add::<LightingPass, LightingOutputs, _>(|pool, store| {
        let gbuffer = *store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let shadow = *store
            .outputs::<ShadowOutputs>()
            .expect("Shadow pass missing");
        let pass = LightingPass::new(
            pool,
            gbuffer,
            shadow,
            size.width,
            size.height,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    let mut ssgi_final: Option<ResourceId> = None;
    if toggles.ssgi_pass {
        let history_id = ssgi_history.expect("ssgi history id missing");

        builder.add::<DownsamplePass, DownsampleOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let lighting = store
                .outputs::<LightingOutputs>()
                .expect("Lighting pass missing");
            let pass = DownsamplePass::new(
                pool,
                gbuffer,
                lighting.lighting_diffuse,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });

        builder.add::<SsgiPass, SsgiOutputs, _>(|pool, store| {
            let downsample = *store
                .outputs::<DownsampleOutputs>()
                .expect("Downsample pass missing");
            let pass = SsgiPass::new(
                pool,
                downsample,
                history_id,
                size.width,
                size.height,
                params.blue_noise_view.clone(),
                params.blue_noise_sampler.clone(),
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });

        let denoised_input = if toggles.ssgi_denoise_pass {
            let denoised = builder.add::<SsgiDenoisePass, SsgiDenoiseOutputs, _>(|pool, store| {
                let ssgi_outputs = store.outputs::<SsgiOutputs>().expect("SSGI pass missing");
                let downsample = store
                    .outputs::<DownsampleOutputs>()
                    .expect("Downsample pass missing");
                let pass = SsgiDenoisePass::new(
                    pool,
                    ssgi_outputs.raw_half,
                    downsample.depth,
                    downsample.normal,
                    history_id,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            });
            Some(denoised.outputs.denoised_half)
        } else {
            None
        };

        let upsample =
            builder.add::<SsgiUpsamplePass, SsgiUpsampleOutputs, _>(move |pool, store| {
                let gbuffer = store
                    .outputs::<GBufferOutputs>()
                    .expect("G-buffer pass missing");
                let source = denoised_input.unwrap_or_else(|| {
                    store
                        .outputs::<SsgiOutputs>()
                        .expect("SSGI pass missing")
                        .raw_half
                });
                let pass = SsgiUpsamplePass::new(
                    pool,
                    source,
                    gbuffer.depth_copy,
                    gbuffer.normal,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            });
        ssgi_final = Some(upsample.outputs.upsampled);
    }

    let mut ssr_output = None;
    if toggles.ssr_pass {
        let ssr = builder.add::<SsrPass, SsrOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let lighting = store
                .outputs::<LightingOutputs>()
                .expect("Lighting pass missing");
            let pass = SsrPass::new(
                pool,
                gbuffer,
                lighting.lighting,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
        ssr_output = Some(ssr.outputs.reflection);
    }

    let ssgi_tex = ssgi_final;
    let ssr_tex = ssr_output;
    let composite =
        builder.add::<DebugCompositePass, DebugCompositeInputs, _>(move |pool, store| {
            let lighting = *store
                .outputs::<LightingOutputs>()
                .expect("Lighting pass missing");
            let ssgi =
                ssgi_tex.or_else(|| store.outputs::<SsgiUpsampleOutputs>().map(|o| o.upsampled));
            let ssr = ssr_tex.or_else(|| store.outputs::<SsrOutputs>().map(|o| o.reflection));
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let sky = *store.outputs::<SkyOutputs>().expect("Sky pass missing");
            let pass = DebugCompositePass::new(
                pool,
                lighting.lighting,
                ssgi,
                ssr,
                gbuffer,
                sky.sky,
                params.surface_format,
            );
            let outputs = pass.inputs();
            (pass, outputs)
        });
    let swapchain_id = composite.outputs.swapchain;

    if toggles.egui_pass {
        builder.add::<EguiPass, EguiOutputs, _>(|pool, _store| {
            let pass = EguiPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    let (graph, passes) = builder.into_parts();
    let mut resource_ids: Vec<ResourceId> = passes.resource_ids().collect();
    if let Some(history) = ssgi_history {
        resource_ids.push(history);
    }
    resource_ids.sort_by_key(|id| id.raw());
    resource_ids.dedup_by_key(|id| id.raw());

    RenderGraphBuildOutput {
        graph,
        swapchain_id,
        resource_ids,
        hiz_id,
    }
}

fn build_traced_graph(
    params: &RenderGraphBuildParams,
    pool: &mut GpuResourcePool,
) -> RenderGraphBuildOutput {
    let size: PhysicalSize<u32> = params.surface_size;
    let toggles = params.config;

    let mut builder = PassGraphBuilder::new(pool);

    builder.add::<RayTracingPass, RayTracingOutputs, _>(|pool, _| {
        let pass = RayTracingPass::new(pool, size.width, size.height);
        let outputs = pass.outputs();
        (pass, outputs)
    });

    let composite =
        builder.add::<RayTracingCompositePass, RayTracingCompositeInputs, _>(|pool, store| {
            let traced = store
                .outputs::<RayTracingOutputs>()
                .expect("Ray tracing pass missing");
            let pass =
                RayTracingCompositePass::new(pool, traced.accumulation, params.surface_format);
            let outputs = pass.inputs();
            (pass, outputs)
        });
    let swapchain_id = composite.outputs.swapchain;

    if toggles.egui_pass {
        builder.add::<EguiPass, EguiOutputs, _>(|pool, _store| {
            let pass = EguiPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    let (graph, passes) = builder.into_parts();
    let mut resource_ids: Vec<ResourceId> = passes.resource_ids().collect();
    resource_ids.sort_by_key(|id| id.raw());
    resource_ids.dedup_by_key(|id| id.raw());

    RenderGraphBuildOutput {
        graph,
        swapchain_id,
        resource_ids,
        hiz_id: None,
    }
}
