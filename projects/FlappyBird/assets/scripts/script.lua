-- Procedural Flappy Bird with an orbit camera
-- Controls: Space / Left Mouse / A or B to flap, R or Y (controller) to reset
-- Camera: mouse move or left stick, wheel/triggers to zoom

local bird = {
    pos = { x = 0.0, y = 0.0, z = 0.0 },
    vel = 0.0,
    scale = { x = 0.8, y = 0.6, z = 0.6 },
    radius = 0.45,
}

local camera = {
    entity = nil,
    yaw = -math.pi * 0.5,
    pitch = 0.35,
    distance = 9.0,
    min_distance = 6.0,
    max_distance = 14.0,
    follow = { x = 0.0, y = 0.0, z = 0.0 },
    yaw_smooth = 8.0,
    pitch_smooth = 8.0,
    follow_smooth = 10.0,
    height_bias = 0.8,
    target_lead = 1.5,
}

-- Tuning: pipe density and pacing
local pipe_spawn_ahead = 8 -- how many pipe pairs stay ahead of the bird
local pipe_start_offset = 12.0 -- distance in front of the bird to start the first pipe
local pipe_despawn_behind = 12.0 -- distance behind the bird before old pipes are removed
local pipe_spacing = 6.0 -- space between pipe centers along the x-axis
local pipe_speed = 5.0 -- how fast pipes move toward the bird
local pipe_gap = 3.5 -- opening size between top/bottom pipes
local pipe_width = 1.4 -- pipe thickness on the x-axis
local pipe_depth = 1.6 -- pipe thickness on the z-axis

-- Tuning: bird physics
local gravity = -18.0
local flap_velocity = 7.5
local ground_y = -3.0
local ceiling_y = 6.0

local pipes = {}
local pipe_serial = 0
local ground_id = nil
local directional_light_id = nil
local spawned_ground = false
local spawned_camera = false

local score = 0
local best = 0
local alive = true

local SUNSET_OFFSETS = {
    { yaw = -0.55, pitch = 0.12, weight = 6 },
    { yaw = 0.55, pitch = 0.12, weight = 6 },
    { yaw = -0.9, pitch = 0.18, weight = 3 },
    { yaw = 0.9, pitch = 0.18, weight = 3 },
    { yaw = -1.2, pitch = 0.24, weight = 2 },
    { yaw = 1.2, pitch = 0.24, weight = 2 },
}

local SUNSET_FALLBACK = {
    { yaw = -2.35, pitch = 0.12, weight = 6 },
    { yaw = 2.35, pitch = 0.12, weight = 6 },
    { yaw = -2.0, pitch = 0.18, weight = 3 },
    { yaw = 2.0, pitch = 0.18, weight = 3 },
    { yaw = -1.7, pitch = 0.24, weight = 2 },
    { yaw = 1.7, pitch = 0.24, weight = 2 },
}

local function clamp(value, min_value, max_value)
    if value < min_value then return min_value end
    if value > max_value then return max_value end
    return value
end

local function lerp(a, b, t)
    return a + (b - a) * t
end

local function smooth(current, target, speed, dt)
    local t = 1.0 - math.exp(-speed * dt)
    return current + (target - current) * t
end

local function vec3_sub(a, b)
    return { x = a.x - b.x, y = a.y - b.y, z = a.z - b.z }
end

local function vec3_mul(a, s)
    return { x = a.x * s, y = a.y * s, z = a.z * s }
end

local function vec3_len(a)
    return math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
end

local function vec3_norm(a)
    local len = vec3_len(a)
    if len == 0.0 then
        return { x = 0.0, y = 0.0, z = 0.0 }
    end
    return vec3_mul(a, 1.0 / len)
end

local function quat_from_yaw_pitch(yaw, pitch)
    local cy = math.cos(yaw * 0.5)
    local sy = math.sin(yaw * 0.5)
    local cp = math.cos(pitch * 0.5)
    local sp = math.sin(pitch * 0.5)
    return { x = sp * cy, y = sy * cp, z = -sy * sp, w = cy * cp }
end

local function quat_from_roll(roll)
    local s = math.sin(roll * 0.5)
    local c = math.cos(roll * 0.5)
    return { x = 0.0, y = 0.0, z = s, w = c }
end

local function random_gap_y()
    local min_gap = ground_y + pipe_gap * 0.5 + 0.5
    local max_gap = ceiling_y - pipe_gap * 0.5 - 0.5
    return lerp(min_gap, max_gap, math.random())
end

local function pick_weighted(list)
    local total = 0
    for i = 1, #list do
        total = total + list[i].weight
    end

    local roll = math.random() * total
    for i = 1, #list do
        roll = roll - list[i].weight
        if roll <= 0.0 then
            return list[i]
        end
    end

    return list[1]
end

local function find_directional_light()
    local entities = ecs.list_entities()
    local fallback = nil
    for i = 1, #entities do
        local id = entities[i]
        local light = ecs.get_light(id)
        if light ~= nil and light.type == "Directional" then
            local name = ecs.get_entity_name(id)
            if name ~= nil then
                local lower = string.lower(name)
                if string.find(lower, "sun", 1, true) or string.find(lower, "directional", 1, true) then
                    return id
                end
            end
            if fallback == nil then
                fallback = id
            end
        end
    end
    return fallback
end

local function pick_preferred_sun_angle(base_yaw)
    local list = SUNSET_FALLBACK
    if base_yaw ~= nil then
        list = SUNSET_OFFSETS
    end

    local choice = pick_weighted(list)
    local yaw = choice.yaw
    if base_yaw ~= nil then
        yaw = base_yaw + choice.yaw
    end
    yaw = yaw + (math.random() * 2.0 - 1.0) * 0.12

    local pitch = choice.pitch + (math.random() * 2.0 - 1.0) * 0.03
    pitch = clamp(pitch, 0.06, 0.28)

    return yaw, pitch
end

local function rotate_directional_light()
    if directional_light_id == nil or not ecs.entity_exists(directional_light_id) then
        directional_light_id = find_directional_light()
    end
    if directional_light_id == nil then
        return
    end

    local transform = ecs.get_transform(directional_light_id)
    if transform == nil then
        return
    end

    local base_yaw = camera and camera.yaw or nil
    if base_yaw ~= nil then
        base_yaw = base_yaw + math.pi
    end

    local yaw, pitch = pick_preferred_sun_angle(base_yaw)
    transform.rotation = quat_from_yaw_pitch(yaw, -pitch)
    ecs.set_transform(directional_light_id, transform)
end

local function update_pipe_transform(pipe)
    local gap_bottom = pipe.gap - pipe_gap * 0.5
    local gap_top = pipe.gap + pipe_gap * 0.5

    local bottom_height = gap_bottom - ground_y
    local top_height = ceiling_y - gap_top

    bottom_height = math.max(bottom_height, 0.1)
    top_height = math.max(top_height, 0.1)

    local bottom_center = ground_y + bottom_height * 0.5
    local top_center = gap_top + top_height * 0.5

    ecs.set_transform(pipe.bottom, {
        position = { x = pipe.x, y = bottom_center, z = 0.0 },
        scale = { x = pipe_width, y = bottom_height, z = pipe_depth },
    })

    ecs.set_transform(pipe.top, {
        position = { x = pipe.x, y = top_center, z = 0.0 },
        scale = { x = pipe_width, y = top_height, z = pipe_depth },
    })
end

local function spawn_pipe_at(x)
    pipe_serial = pipe_serial + 1
    local top = ecs.spawn_entity("Pipe Top " .. tostring(pipe_serial))
    local bottom = ecs.spawn_entity("Pipe Bottom " .. tostring(pipe_serial))

    ecs.set_mesh_renderer(top, {
        source = "Cube",
        material = "assets/materials/pipe.ron",
        casts_shadow = true,
        visible = true,
    })

    ecs.set_mesh_renderer(bottom, {
        source = "Cube",
        material = "assets/materials/pipe.ron",
        casts_shadow = true,
        visible = true,
    })

    local pipe = {
        top = top,
        bottom = bottom,
        x = x,
        gap = random_gap_y(),
        scored = false,
    }

    table.insert(pipes, pipe)
    update_pipe_transform(pipe)
end

local function cleanup_pipes()
    for i = 1, #pipes do
        local pipe = pipes[i]
        ecs.delete_entity(pipe.top)
        ecs.delete_entity(pipe.bottom)
    end
    pipes = {}
    pipe_serial = 0
end

local function reset_pipes()
    cleanup_pipes()

    local start_x = bird.pos.x + pipe_start_offset
    for i = 1, pipe_spawn_ahead do
        spawn_pipe_at(start_x + (i - 1) * pipe_spacing)
    end
end

local function ensure_pipe_buffer()
    local target_x = bird.pos.x + pipe_spawn_ahead * pipe_spacing
    local farthest_x = bird.pos.x + pipe_start_offset - pipe_spacing

    for i = 1, #pipes do
        if pipes[i].x > farthest_x then
            farthest_x = pipes[i].x
        end
    end

    while farthest_x < target_x do
        farthest_x = farthest_x + pipe_spacing
        spawn_pipe_at(farthest_x)
    end
end

local function reset_camera_state()
    camera.follow = { x = bird.pos.x, y = bird.pos.y, z = bird.pos.z }
    camera.yaw = -math.pi * 0.5
    camera.pitch = 0.35

    if camera.entity ~= nil then
        ecs.set_active_camera(camera.entity)
    end
end

local function reset_game()
    bird.pos = { x = 0.0, y = 0.0, z = 0.0 }
    bird.vel = 0.0

    score = 0
    alive = true

    rotate_directional_light()
    reset_pipes()
    reset_camera_state()
end

local function crash()
    if not alive then return true end
    alive = false
    if score > best then
        best = score
    end
    print("Crashed! score: " .. tostring(score) .. " best: " .. tostring(best) .. " (respawning)")
    reset_game()
    return true
end

local function update_bird_transform()
    local roll = clamp(bird.vel * 0.08, -0.9, 0.6)
    ecs.set_transform(entity_id, {
        position = { x = bird.pos.x, y = bird.pos.y, z = bird.pos.z },
        rotation = quat_from_roll(roll),
        scale = bird.scale,
    })
end

local function update_camera(dt)
    if camera.entity == nil then return end

    -- Zoom with wheel or triggers
    local wheel = input.wheel()
    camera.distance = clamp(camera.distance - wheel.y * 0.8, camera.min_distance, camera.max_distance)

    local using_stick = false
    if input.gamepad_count() > 0 then
        local ids = input.gamepad_ids()
        local gamepad_id = ids[1]
        local deadzone = 0.12
        local look_speed = 2.4
        local zoom_speed = 8.0

        local sx = input.gamepad_axis(input.gamepad_axes.LeftX, gamepad_id)
        local sy = input.gamepad_axis(input.gamepad_axes.LeftY, gamepad_id)
        if math.abs(sx) > deadzone or math.abs(sy) > deadzone then
            camera.yaw = camera.yaw + sx * look_speed * dt
            camera.pitch = clamp(camera.pitch + sy * look_speed * dt, 0.05, 1.15)
            using_stick = true
        end

        local zoom_in = input.gamepad_trigger("right", gamepad_id)
        local zoom_out = input.gamepad_trigger("left", gamepad_id)
        local zoom_delta = (zoom_out - zoom_in) * zoom_speed * dt
        if math.abs(zoom_delta) > 0.0001 then
            camera.distance = clamp(camera.distance + zoom_delta, camera.min_distance, camera.max_distance)
        end
    end

    -- Mouse orbit when the stick is idle
    if not using_stick and not input.wants_pointer() then
        local size = input.window_size()
        local cursor = input.cursor()
        local nx = 0.5
        local ny = 0.5

        if size.x > 1.0 then
            nx = clamp(cursor.x / size.x, 0.0, 1.0)
        end
        if size.y > 1.0 then
            ny = clamp(cursor.y / size.y, 0.0, 1.0)
        end

        local behind_yaw = -math.pi * 0.5
        local side_span = math.pi * 0.5
        local target_yaw = behind_yaw + (nx * 2.0 - 1.0) * side_span
        local pitch_min = 0.1
        local pitch_max = 0.55
        local target_pitch = lerp(pitch_max, pitch_min, ny)

        camera.yaw = smooth(camera.yaw, target_yaw, camera.yaw_smooth, dt)
        camera.pitch = smooth(camera.pitch, target_pitch, camera.pitch_smooth, dt)
    end

    camera.follow.x = smooth(camera.follow.x, bird.pos.x + camera.target_lead, camera.follow_smooth, dt)
    camera.follow.y = smooth(camera.follow.y, bird.pos.y + 0.2, camera.follow_smooth, dt)
    camera.follow.z = smooth(camera.follow.z, bird.pos.z, camera.follow_smooth, dt)

    local cos_pitch = math.cos(camera.pitch)
    local offset = {
        x = math.sin(camera.yaw) * cos_pitch * camera.distance,
        y = math.sin(camera.pitch) * camera.distance,
        z = math.cos(camera.yaw) * cos_pitch * camera.distance,
    }

    local cam_pos = {
        x = camera.follow.x + offset.x,
        y = camera.follow.y + offset.y + camera.height_bias,
        z = camera.follow.z + offset.z,
    }

    local target = {
        x = camera.follow.x + camera.target_lead,
        y = camera.follow.y,
        z = camera.follow.z,
    }

    local dir = vec3_norm(vec3_sub(target, cam_pos))
    local yaw = math.atan2(dir.x, dir.z)
    local pitch = math.asin(-dir.y)
    local rot = quat_from_yaw_pitch(yaw, pitch)

    ecs.set_transform(camera.entity, {
        position = cam_pos,
        rotation = rot,
    })
end

local function handle_input(dt)
    local flap =
        input.key_pressed(input.keys.Space)
        or input.mouse_pressed(input.mouse_buttons.Left)
        or input.gamepad_button_pressed(input.gamepad_buttons.South)
        or input.gamepad_button_pressed(input.gamepad_buttons.East)

    if flap then
        bird.vel = flap_velocity
    end

    if input.key_pressed(input.keys.R) or input.gamepad_button_pressed(input.gamepad_buttons.North) then
        reset_game()
    end

    bird.vel = bird.vel + gravity * dt
    bird.pos.y = bird.pos.y + bird.vel * dt

    if bird.pos.y - bird.radius < ground_y then
        bird.pos.y = ground_y + bird.radius
        crash()
        return
    elseif bird.pos.y + bird.radius > ceiling_y then
        bird.pos.y = ceiling_y - bird.radius
        crash()
        return
    end
end

local function update_pipes(dt)
    for i = #pipes, 1, -1 do
        local pipe = pipes[i]
        pipe.x = pipe.x - pipe_speed * dt

        if pipe.x < bird.pos.x - pipe_despawn_behind then
            ecs.delete_entity(pipe.top)
            ecs.delete_entity(pipe.bottom)
            table.remove(pipes, i)
        else
            update_pipe_transform(pipe)

            if alive then
                local half_width = pipe_width * 0.5 + bird.radius
                if math.abs(pipe.x - bird.pos.x) < half_width then
                    local gap_bottom = pipe.gap - pipe_gap * 0.5
                    local gap_top = pipe.gap + pipe_gap * 0.5
                    if bird.pos.y - bird.radius < gap_bottom or bird.pos.y + bird.radius > gap_top then
                        crash()
                        return
                    end
                end

                if not pipe.scored and (pipe.x + pipe_width * 0.5) < bird.pos.x then
                    pipe.scored = true
                    score = score + 1
                    if score > best then
                        best = score
                    end
                    print("Score: " .. tostring(score))
                end
            end
        end
    end

    ensure_pipe_buffer()
end

function on_start()
    math.randomseed(os.time() % 100000)

    if not ecs.has_component(entity_id, "mesh_renderer") then
        ecs.set_mesh_renderer(entity_id, {
            source = "Cube",
            material = "assets/materials/bird.ron",
            casts_shadow = true,
            visible = true,
        })
    end

    if camera.entity == nil then
        camera.entity = ecs.spawn_entity("Bird Camera")
        spawned_camera = true
        ecs.set_camera(camera.entity, {
            fov_y_rad = math.rad(60.0),
            aspect_ratio = 1.777,
            near_plane = 0.05,
            far_plane = 200.0,
            active = true,
        })
    end

    if ground_id == nil then
        ground_id = ecs.spawn_entity("Ground")
        spawned_ground = true
        ecs.set_mesh_renderer(ground_id, {
            source = "Cube",
            material = "assets/materials/ground.ron",
            casts_shadow = true,
            visible = true,
        })
        ecs.set_transform(ground_id, {
            position = { x = 0.0, y = ground_y - 0.5, z = 0.0 },
            scale = { x = 80.0, y = 1.0, z = 20.0 },
        })
    end

    reset_game()
    update_bird_transform()
end

function on_update(dt)
    handle_input(dt)
    update_pipes(dt)
    update_bird_transform()
    update_camera(dt)
end

function on_stop()
    cleanup_pipes()

    if spawned_camera and camera.entity ~= nil then
        ecs.delete_entity(camera.entity)
    end

    if spawned_ground and ground_id ~= nil then
        ecs.delete_entity(ground_id)
    end

    camera.entity = nil
    ground_id = nil
end
