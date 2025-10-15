# [helmer](http://helmer.leighteg.dev) &nbsp; ![Status: Pre-Alpha](https://img.shields.io/badge/status-Pre--Alpha-orange)

an ECS game engine featuring 3 purpose built renderers written in rust

![city corner screenshot (deferred renderer)](assets/screenshots/city_corner.png)

## features
- multiplatform
- large scale hardware/driver support
- performant multithreading
- "bring your own logic" approach providing callbacks to implement a custom logic system or ecs
- custom (naive but flexible) ecs implementation/integration, bevy_ecs integration
- rapier3D physics integration
- basic pbr (full pbr todo)
- right handed +Y-up coordinate system
- deferred bindless renderer, forward per-material uniforms renderer, forward texture array renderer
- sharp cascaded EVSM shadows
- SSGI, SSR, physically based sky (deferred only rn)
- frustum culling
- LOD generation

## todo 😡
- [x] provide custom runtime hooks?
- [ ] provide custom renderer hooks
- [x] provide custom logic thread hooks
- [x] "bring your own logic" option (through logic thread hooks)
- [ ] ecs revamp ❓
- [ ] custom types to relieve project reliance on engine dependencies (InputMan)
- [x] ui implementation or integration
- [ ] editor & project cli tools
- [ ] taskable worker pool system/api for ecs (resource based? - asynchronously taskable by systems?)
- [ ] asset streaming
- [ ] AssetServer improved worker pool usage (scene parsing)
- [ ] skinned mesh & animation system/support
- [ ] precomputed atmospheric scattering
- [ ] fix sky light contribution
- [ ] account for atmosphere ground
- [ ] SSR cubemap fallback?
- [ ] more light types
- [ ] occlusion culling
- [ ] soft shadows (PCSS?)
- [ ] point light shadows
- [ ] full gi (SDFGI, DDGI?)
- [ ] full pbr/modern brdf
- [ ] implement supported advanced features to forward renderers
- [ ] gpu compute based culling?
- [ ] hardware RT path/pipeline or acceleration?
- [ ] touch support
- [ ] physically based audio engine