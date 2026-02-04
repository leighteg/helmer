use winit::dpi::PhysicalSize;

use crate::graphics::{
    common::{
        config::{
            DEBUG_FLAG_DIRECT, DEBUG_FLAG_EMISSION, DEBUG_FLAG_INDIRECT, DEBUG_FLAG_REFLECTIONS,
            DEBUG_FLAG_SKY, DEBUG_FLAG_UNLIT, RenderConfig,
        },
        graph::{RenderGraphBuildOutput, RenderGraphBuildParams, RenderGraphSpec},
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
        ddgi_probe_update::{DdgiProbeOutputs, DdgiProbeUpdatePass},
        ddgi_resample::{DdgiResampleOutputs, DdgiResamplePass},
        debug_composite::{DebugCompositeInputs, DebugCompositePass},
        depth_copy::{DepthCopyOutputs, DepthCopyPass},
        downsample::{DownsampleOutputs, DownsamplePass},
        egui::{EguiOutputs, EguiPass},
        forward::{ForwardOutputs, ForwardPass},
        gbuffer::{GBufferFormats, GBufferOutputs, GBufferPass},
        gizmo::{GizmoOutputs, GizmoPass},
        hiz::{HiZOutputs, HiZPass},
        lighting::{LightingOutputs, LightingPass},
        raytraced::{
            raytracing::{RayTracingOutputs, RayTracingPass},
            raytracing_composite::{RayTracingCompositeInputs, RayTracingCompositePass},
        },
        reflection_combine::{ReflectionCombineOutputs, ReflectionCombinePass},
        rt_reflections::{RtReflectionsOutputs, RtReflectionsPass},
        rt_reflections_denoise::{RtReflectionsDenoiseOutputs, RtReflectionsDenoisePass},
        shadow::{ShadowOutputs, ShadowPass},
        sky::{SkyOutputs, SkyPass},
        ssgi::{SsgiOutputs, SsgiPass},
        ssgi_denoise::{SsgiDenoiseOutputs, SsgiDenoisePass},
        ssgi_upsample::{SsgiUpsampleOutputs, SsgiUpsamplePass},
        ssr::{SsrOutputs, SsrPass},
    },
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
    Ddgi,
    Egui,
    Gizmo,
    Occlusion,
    Transparent,
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
            RenderPassToggleFlag::Ddgi => config.ddgi_pass,
            RenderPassToggleFlag::Egui => config.egui_pass,
            RenderPassToggleFlag::Gizmo => config.gizmo_pass,
            RenderPassToggleFlag::Occlusion => config.occlusion_culling,
            RenderPassToggleFlag::Transparent => config.transparent_pass,
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
            RenderPassToggleFlag::Ddgi => config.ddgi_pass = value,
            RenderPassToggleFlag::Egui => config.egui_pass = value,
            RenderPassToggleFlag::Gizmo => config.gizmo_pass = value,
            RenderPassToggleFlag::Occlusion => config.occlusion_culling = value,
            RenderPassToggleFlag::Transparent => config.transparent_pass = value,
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
        label: "Transparency",
        toggle: RenderPassToggleFlag::Transparent,
    },
    RenderPassToggle {
        label: "Egui",
        toggle: RenderPassToggleFlag::Egui,
    },
    RenderPassToggle {
        label: "Gizmo",
        toggle: RenderPassToggleFlag::Gizmo,
    },
    RenderPassToggle {
        label: "Occlusion (Hi-Z)",
        toggle: RenderPassToggleFlag::Occlusion,
    },
];

const HYBRID_GRAPH_PASSES: &[RenderPassToggle] = &[
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
        label: "DDGI Resampling",
        toggle: RenderPassToggleFlag::Ddgi,
    },
    RenderPassToggle {
        label: "Transparency",
        toggle: RenderPassToggleFlag::Transparent,
    },
    RenderPassToggle {
        label: "Egui",
        toggle: RenderPassToggleFlag::Egui,
    },
    RenderPassToggle {
        label: "Gizmo",
        toggle: RenderPassToggleFlag::Gizmo,
    },
    RenderPassToggle {
        label: "Occlusion (Hi-Z)",
        toggle: RenderPassToggleFlag::Occlusion,
    },
];

const DEBUG_GRAPH_PASSES: &[RenderPassToggle] = &[
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
        label: "Transparency",
        toggle: RenderPassToggleFlag::Transparent,
    },
    RenderPassToggle {
        label: "Egui",
        toggle: RenderPassToggleFlag::Egui,
    },
    RenderPassToggle {
        label: "Gizmo",
        toggle: RenderPassToggleFlag::Gizmo,
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

const TRACED_GRAPH_PASSES: &[RenderPassToggle] = &[
    RenderPassToggle {
        label: "Egui",
        toggle: RenderPassToggleFlag::Egui,
    },
    RenderPassToggle {
        label: "Gizmo",
        toggle: RenderPassToggleFlag::Gizmo,
    },
];

const GRAPH_TEMPLATES: &[RenderGraphTemplate] = &[
    RenderGraphTemplate {
        name: "default-graph",
        label: "Default Graph",
        build: default_graph_spec,
        pass_toggles: DEFAULT_GRAPH_PASSES,
        debug_flags: &[],
    },
    RenderGraphTemplate {
        name: "hybrid-graph",
        label: "Hybrid Graph",
        build: hybrid_graph_spec,
        pass_toggles: HYBRID_GRAPH_PASSES,
        debug_flags: &[],
    },
    RenderGraphTemplate {
        name: "debug-graph",
        label: "Debug Graph",
        build: debug_graph_spec,
        pass_toggles: DEBUG_GRAPH_PASSES,
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

impl PassResourceOutput for DdgiResampleOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.diffuse, self.specular]
    }
}

impl PassResourceOutput for DdgiProbeOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![
            self.irradiance_a,
            self.irradiance_b,
            self.distance_a,
            self.distance_b,
        ]
    }
}

impl PassResourceOutput for RtReflectionsOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.reflection]
    }
}

impl PassResourceOutput for RtReflectionsDenoiseOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.reflection]
    }
}

impl PassResourceOutput for ReflectionCombineOutputs {
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

impl PassResourceOutput for ForwardOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.swapchain]
    }
}

impl PassResourceOutput for EguiOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.swapchain]
    }
}

impl PassResourceOutput for GizmoOutputs {
    fn resource_ids(&self) -> Vec<ResourceId> {
        vec![self.swapchain]
    }
}

fn depth_copy_output_is_single_channel(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::R32Float | wgpu::TextureFormat::R16Float
    )
}

fn select_gbuffer_formats(params: &RenderGraphBuildParams) -> GBufferFormats {
    let mut formats = if params.force_low_color_formats {
        GBufferFormats::ultra()
    } else {
        GBufferFormats::select(
            params
                .device_caps
                .limits
                .max_color_attachment_bytes_per_sample,
        )
    };

    formats.depth = params.depth_format;
    formats.depth_copy = params.depth_copy_format;

    let required_attachments = if formats.output_depth_copy { 5 } else { 4 };
    if params.max_color_attachments < required_attachments
        || !depth_copy_output_is_single_channel(formats.depth_copy)
    {
        formats.output_depth_copy = false;
    }

    formats
}

/// Default graph used by the renderer: shadow -> gbuffer -> lighting -> post.
pub fn default_graph_spec() -> RenderGraphSpec {
    RenderGraphSpec::unique("default-graph", |params, pool| {
        build_default_graph(params, pool)
    })
}

/// Hybrid graph: raster passes w/ ray-traced DDGI resampling.
pub fn hybrid_graph_spec() -> RenderGraphSpec {
    RenderGraphSpec::unique("hybrid-graph", |params, pool| {
        build_hybrid_graph(params, pool)
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

    let gbuffer_formats = select_gbuffer_formats(params);
    let use_uniform_materials = !params.supports_fragment_storage_buffers;
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
            use_uniform_materials,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if !gbuffer_formats.output_depth_copy {
        builder.add::<DepthCopyPass, DepthCopyOutputs, _>(|_, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = DepthCopyPass::new(
                gbuffer.depth,
                gbuffer.depth_copy,
                size.width,
                size.height,
                gbuffer_formats.depth_copy,
            );
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            !params.device_caps.supports_compute(),
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            params.supports_fragment_storage_buffers,
            !params.device_caps.supports_compute(),
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

    if toggles.transparent_pass {
        builder.add::<ForwardPass, ForwardOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let shadow = *store
                .outputs::<ShadowOutputs>()
                .expect("Shadow pass missing");
            let pass = ForwardPass::new(
                pool,
                swapchain_id,
                gbuffer.depth,
                shadow.map,
                params.surface_format,
                params.supports_fragment_storage_buffers,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    if toggles.gizmo_pass {
        builder.add::<GizmoPass, GizmoOutputs, _>(|pool, _store| {
            let pass = GizmoPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

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

fn build_hybrid_graph(
    params: &RenderGraphBuildParams,
    pool: &mut GpuResourcePool,
) -> RenderGraphBuildOutput {
    let size: PhysicalSize<u32> = params.surface_size;
    let toggles = params.config;
    let mut hiz_id = None;

    let gbuffer_formats = select_gbuffer_formats(params);
    let use_uniform_materials = !params.supports_fragment_storage_buffers;
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
            use_uniform_materials,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if !gbuffer_formats.output_depth_copy {
        builder.add::<DepthCopyPass, DepthCopyOutputs, _>(|_, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = DepthCopyPass::new(
                gbuffer.depth,
                gbuffer.depth_copy,
                size.width,
                size.height,
                gbuffer_formats.depth_copy,
            );
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            !params.device_caps.supports_compute(),
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            params.supports_fragment_storage_buffers,
            !params.device_caps.supports_compute(),
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    let mut ddgi_diffuse = None;
    let mut ddgi_specular = None;
    if toggles.ddgi_pass {
        let mut counts = [
            params.config.ddgi_probe_count_x.max(1),
            params.config.ddgi_probe_count_y.max(1),
            params.config.ddgi_probe_count_z.max(1),
        ];
        let mut probe_resolution = params
            .config
            .ddgi_probe_resolution
            .max(1)
            .min(params.device_caps.limits.max_texture_dimension_2d.max(1));
        if probe_resolution == 0 {
            probe_resolution = 1;
        }
        let max_layers = params.device_caps.limits.max_texture_array_layers.max(1);
        let mut total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
        if total > max_layers as u64 {
            let scale = (max_layers as f64 / total as f64).cbrt();
            if scale.is_finite() && scale > 0.0 {
                counts[0] = ((counts[0] as f64 * scale).floor() as u32).max(1);
                counts[1] = ((counts[1] as f64 * scale).floor() as u32).max(1);
                counts[2] = ((counts[2] as f64 * scale).floor() as u32).max(1);
            }
            total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
            while total > max_layers as u64 {
                let mut largest = 0usize;
                if counts[1] > counts[largest] {
                    largest = 1;
                }
                if counts[2] > counts[largest] {
                    largest = 2;
                }
                if counts[largest] <= 1 {
                    break;
                }
                counts[largest] -= 1;
                total = counts[0] as u64 * counts[1] as u64 * counts[2] as u64;
            }
        }

        let probe_count = total.max(1) as u32;
        let probe_update = builder.add::<DdgiProbeUpdatePass, DdgiProbeOutputs, _>(|pool, _| {
            let pass = DdgiProbeUpdatePass::new(pool, probe_resolution, probe_count);
            let outputs = pass.outputs();
            (pass, outputs)
        });

        let ddgi = builder.add::<DdgiResamplePass, DdgiResampleOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let probes = *store
                .outputs::<DdgiProbeOutputs>()
                .expect("DDGI probe pass missing");
            let pass = DdgiResamplePass::new(
                pool,
                probes,
                gbuffer,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
        ddgi_diffuse = Some(ddgi.outputs.diffuse);
        ddgi_specular = Some(ddgi.outputs.specular);
    }

    let mut primary_reflection = None;
    let probe_reflection = ddgi_specular;
    if toggles.ssr_pass && !params.config.rt_reflections {
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
        primary_reflection = Some(ssr.outputs.reflection);
    }

    if params.config.rt_reflections {
        let rt = builder.add::<RtReflectionsPass, RtReflectionsOutputs, _>(move |pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = RtReflectionsPass::new(
                pool,
                gbuffer,
                size.width,
                size.height,
                params.config.use_transient_textures,
                params.config.use_transient_aliasing,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
        let rt_reflection = rt.outputs.reflection;
        let denoise = builder.add::<RtReflectionsDenoisePass, RtReflectionsDenoiseOutputs, _>(
            move |pool, store| {
                let gbuffer = *store
                    .outputs::<GBufferOutputs>()
                    .expect("G-buffer pass missing");
                let pass = RtReflectionsDenoisePass::new(
                    pool,
                    rt_reflection,
                    gbuffer,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            },
        );
        primary_reflection = Some(denoise.outputs.reflection);
    }

    let mut combined_reflection = None;
    if let (Some(primary), Some(probe)) = (primary_reflection, probe_reflection) {
        let combine = builder.add::<ReflectionCombinePass, ReflectionCombineOutputs, _>(
            move |pool, _store| {
                let pass = ReflectionCombinePass::new(
                    pool,
                    primary,
                    probe,
                    size.width,
                    size.height,
                    params.config.use_transient_textures,
                    params.config.use_transient_aliasing,
                );
                let outputs = pass.outputs();
                (pass, outputs)
            },
        );
        combined_reflection = Some(combine.outputs.reflection);
    } else if primary_reflection.is_some() {
        combined_reflection = primary_reflection;
    } else if probe_reflection.is_some() {
        combined_reflection = probe_reflection;
    }

    let ddgi_tex = ddgi_diffuse;
    let reflection_tex = combined_reflection;
    let composite = builder.add::<CompositePass, CompositeInputs, _>(move |pool, store| {
        let lighting = *store
            .outputs::<LightingOutputs>()
            .expect("Lighting pass missing");
        let gbuffer = *store
            .outputs::<GBufferOutputs>()
            .expect("G-buffer pass missing");
        let sky = *store.outputs::<SkyOutputs>().expect("Sky pass missing");
        let pass = CompositePass::new(
            pool,
            lighting.lighting,
            ddgi_tex,
            reflection_tex,
            gbuffer,
            sky.sky,
            params.surface_format,
        );
        let outputs = pass.inputs();
        (pass, outputs)
    });
    let swapchain_id = composite.outputs.swapchain;

    if toggles.transparent_pass {
        builder.add::<ForwardPass, ForwardOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let shadow = *store
                .outputs::<ShadowOutputs>()
                .expect("Shadow pass missing");
            let pass = ForwardPass::new(
                pool,
                swapchain_id,
                gbuffer.depth,
                shadow.map,
                params.surface_format,
                params.supports_fragment_storage_buffers,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    if toggles.gizmo_pass {
        builder.add::<GizmoPass, GizmoOutputs, _>(|pool, _store| {
            let pass = GizmoPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

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

    let gbuffer_formats = select_gbuffer_formats(params);
    let use_uniform_materials = !params.supports_fragment_storage_buffers;
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
            use_uniform_materials,
        );
        let outputs = pass.outputs();
        (pass, outputs)
    });

    if !gbuffer_formats.output_depth_copy {
        builder.add::<DepthCopyPass, DepthCopyOutputs, _>(|_, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let pass = DepthCopyPass::new(
                gbuffer.depth,
                gbuffer.depth_copy,
                size.width,
                size.height,
                gbuffer_formats.depth_copy,
            );
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            !params.device_caps.supports_compute(),
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
            params.hdr_format,
            params.config.use_transient_textures,
            params.config.use_transient_aliasing,
            params.supports_fragment_storage_buffers,
            !params.device_caps.supports_compute(),
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

    if toggles.transparent_pass {
        builder.add::<ForwardPass, ForwardOutputs, _>(|pool, store| {
            let gbuffer = *store
                .outputs::<GBufferOutputs>()
                .expect("G-buffer pass missing");
            let shadow = *store
                .outputs::<ShadowOutputs>()
                .expect("Shadow pass missing");
            let pass = ForwardPass::new(
                pool,
                swapchain_id,
                gbuffer.depth,
                shadow.map,
                params.surface_format,
                params.supports_fragment_storage_buffers,
            );
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

    if toggles.gizmo_pass {
        builder.add::<GizmoPass, GizmoOutputs, _>(|pool, _store| {
            let pass = GizmoPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

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

    if toggles.gizmo_pass {
        builder.add::<GizmoPass, GizmoOutputs, _>(|pool, _store| {
            let pass = GizmoPass::new(pool, swapchain_id, params.surface_format);
            let outputs = pass.outputs();
            (pass, outputs)
        });
    }

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
