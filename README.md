# [helmer](http://helmer.leighteg.dev) &nbsp; ![Status: Pre-Alpha](https://img.shields.io/badge/status-Pre--Alpha-orange)

### this todo really needs a rewrite - the entire repo needs proper docs - but i dont feel like it/cant. this readme sucks

a performant, flexible, extensible, scalable foundation for creation - featuring a robust render graph and thoughtful architecture, allowing completely custom logic integrations. multiple logic integrations are provided.

![city corner screenshot (legacy deferred renderer)](assets/screenshots/city_corner.png)

## features
- multiplatform
- large scale hardware/driver support
- performant multithreading
- "bring your own logic" approach providing callbacks to implement a custom logic system or ecs
- custom (naive but flexible) ecs implementation/integration, bevy_ecs integration
- rapier3D physics integration
- basic pbr (full pbr todo)
- right handed +Y-up coordinate system
- highly capable, scalable, robust render graph
- legacy monolithic renderers: deferred bindless renderer, forward per-material uniforms renderer, forward texture array renderer
- sharp cascaded EVSM shadows
- SSGI, SSR, physically based sky
- frustum+occlusion culling
- LOD/mip generation

## todo 😡
- [x] provide custom runtime hooks?
- [x] ~~provide custom renderer hooks~~ [render graph]
- [x] provide custom logic thread hooks
- [x] "bring your own logic" option (through logic thread hooks)
- [ ] provide integrations to multiple popular and custom logic architectures
- [ ] helmer_ecs revamp ❓
- [ ] custom types to relieve project reliance on engine dependencies (InputMan)
- [x] ui implementation or integration
- [ ] editor & project cli tools
- [ ] taskable worker pool system/api for ecs (resource based? - asynchronously taskable by systems?)
- [x] asset streaming
- [x] AssetServer improved worker pool usage (scene parsing)
- [ ] skinned mesh & animation system/support
- [x] precomputed atmospheric scattering
- [x] fix sky light contribution (generally fixed)
- [x] account for atmosphere ground (we sorta do)
- [ ] modular cubemap generics
- [ ] SSR cubemap fallback
- [ ] more light types
- [x] occlusion culling [broken?]
- [x] soft shadows (PCSS?) [basic pcf for now]
- [ ] point light shadows
- [ ] full gi (SDFGI, DDGI?)
- [ ] full pbr/modern brdf
- [ ] ~~implement supported advanced features to forward renderers~~
- [ ] ~~replace forward renderer(s) shadow pipeline with a simpler implementation~~ [implement alternate shadow pipeline]
- [ ] ~~add indirect lighting to forward renderer(s)? (Simplified Light Propagation Volumes?)~~
- [ ] ~~cubemap-based reflections in forward renderers~~
- [ ] ~~gpu compute based culling?~~
- [ ] hardware RT path/pipeline or acceleration?
- [ ] touch support
- [ ] physically based audio engine