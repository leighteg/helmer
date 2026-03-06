# helmer

the `helmer` crate is the actual runtime. it is pulled in as a dependency by most crates in this repo

as of now, this crate also hosts all logic pertaining to rendering, assets, audio, input, etc..

it is responsible for the creation/management of the logic/render threads. the runtime provides callbacks enabling the end-dev to hook their own logic into the logic thread, but right now this is limited to the logic thread only (the runtime should strictly be a robust vessel/receptacle! yet massive components (which likely should be condensed into their own crates) like `AssetServer`/`AudioEngine` are currently hardcoded, and the render thread both isnt optional & hardcoded as well)

the runtime uses `winit` so we dont have to deal with platform-specific abstraction. the runtime simply implements it's event loop and abstracts all events down to the runtime's components (eg. InputManager). no propagation should need to be done here; we are simply passing state. many events & their artifacts arent robustly exposed as of now

## input manager
abstracts input up from winit events, providing ergonomic APIs for the end-dev to work with

## render graph
a lot to tackle here. kind of like bevy's. but asset streaming is inherent

as of now all graph templates/passes are hardcoded and need to be modularized into their own crates. the graph itself likely should be modularized as well, followed by proper docs

GraphRenderer is the only renderer that makes use of the graph, and it is ridiculously monolithic currently - promotion of statelessness is big todo

## asset server
thread/worker pool responsible for loading/processing all assets.

responsible for all asset derivatives like LODs/cpu mips/etc..

GraphRenderer can/may request cpu mips

## audio engine
a robust spacial audio engine over `cpal`

needs to be restructured its ridiculous