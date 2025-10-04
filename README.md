# helmer
![Status: Pre-Alpha](https://img.shields.io/badge/status-Pre--Alpha-orange)

an ECS game engine featuring 3 purpose built renderers written in rust

![city corner screenshot (deferred renderer)](assets/screenshots/city_corner.png)

## features
- multiplatform
- large scale hardware/driver support
- performant multithreading
- naive but capable full ECS implementation (rewrite or optimize todo)
- rapier3D physics integration
- basic pbr (full pbr todo)
- right handed +Y-up coordinate system
- deferred bindless renderer, forward per-material uniforms renderer, forward texture array renderer
- sharp cascaded EVSM shadows
- SSGI, SSR, physically based sky (deferred only rn)
- frustum culling
- LOD generation

## todo 😡
- [ ] provide custom runtime hooks?
- [ ] provide custom renderer hooks
- [ ] provide custom logic thread hooks
- [ ] "bring your own logic" option (through logic thread hooks)
- [ ] ecs revamp
- [ ] custom types to relieve project reliance on engine dependencies (InputMan)
- [ ] ui implementation or integration
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
- [ ] full pbr/modern brdf
- [ ] implement supported advanced features to forward renderers
- [ ] gpu compute based culling?
- [ ] hardware RT path/pipeline or acceleration?