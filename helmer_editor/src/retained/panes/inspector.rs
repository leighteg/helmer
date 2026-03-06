use bevy_ecs::entity::Entity;
use helmer_ui::{
    RetainedUi, RetainedUiNode, UiButton, UiButtonVariant, UiColor, UiDimension, UiId, UiLabel,
    UiLayoutBuilder, UiRect, UiStyle, UiTextAlign, UiTextField, UiTextStyle, UiTextValue,
    UiVisualStyle, UiWidget,
};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectorLightKind {
    Directional,
    Point,
    Spot,
}

#[derive(Clone, Debug)]
pub struct InspectorPaneData {
    pub entity: Option<Entity>,
    pub entity_label: String,
    pub name_value: String,
    pub has_transform: bool,
    pub transform_position: [f32; 3],
    pub transform_rotation: [f32; 3],
    pub transform_scale: [f32; 3],
    pub has_camera: bool,
    pub camera_active: bool,
    pub camera_fov_deg: f32,
    pub camera_aspect_ratio: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    pub has_light: bool,
    pub light_kind: InspectorLightKind,
    pub light_color: [f32; 3],
    pub light_color_hsv: [f32; 3],
    pub light_color_picker_open: bool,
    pub light_intensity: f32,
    pub light_spot_angle_deg: f32,
    pub has_mesh_renderer: bool,
    pub mesh_id: usize,
    pub material_id: usize,
    pub mesh_casts_shadow: bool,
    pub mesh_visible: bool,
    pub focused_field: Option<InspectorPaneTextField>,
    pub text_cursors: HashMap<InspectorPaneTextField, usize>,
}

impl Default for InspectorPaneData {
    fn default() -> Self {
        Self {
            entity: None,
            entity_label: "<none>".to_string(),
            name_value: String::new(),
            has_transform: false,
            transform_position: [0.0, 0.0, 0.0],
            transform_rotation: [0.0, 0.0, 0.0],
            transform_scale: [1.0, 1.0, 1.0],
            has_camera: false,
            camera_active: false,
            camera_fov_deg: 45.0,
            camera_aspect_ratio: 1.7,
            camera_near: 0.1,
            camera_far: 100.0,
            has_light: false,
            light_kind: InspectorLightKind::Directional,
            light_color: [1.0, 1.0, 1.0],
            light_color_hsv: [0.0, 0.0, 1.0],
            light_color_picker_open: false,
            light_intensity: 1.0,
            light_spot_angle_deg: 45.0,
            has_mesh_renderer: false,
            mesh_id: 0,
            material_id: 0,
            mesh_casts_shadow: true,
            mesh_visible: true,
            focused_field: None,
            text_cursors: HashMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InspectorPaneTextField {
    Name,
    Transform(InspectorTransformField),
    Camera(InspectorCameraField),
    Light(InspectorLightField),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InspectorTransformField {
    PositionX,
    PositionY,
    PositionZ,
    RotationX,
    RotationY,
    RotationZ,
    ScaleX,
    ScaleY,
    ScaleZ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InspectorCameraField {
    FovDeg,
    Aspect,
    Near,
    Far,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InspectorLightField {
    Intensity,
    SpotAngleDeg,
}

#[derive(Clone, Copy, Debug)]
pub enum InspectorPaneAction {
    SetActiveCamera(Entity),
    DeleteEntity(Entity),
    OpenAddComponentMenu(Entity),
    RemoveTransform(Entity),
    RemoveCamera(Entity),
    RemoveLight(Entity),
    RemoveMeshRenderer(Entity),
    AdjustTransform {
        entity: Entity,
        field: InspectorTransformField,
        delta: f32,
    },
    AdjustCamera {
        entity: Entity,
        field: InspectorCameraField,
        delta: f32,
    },
    SetLightType {
        entity: Entity,
        kind: InspectorLightKind,
    },
    AdjustLight {
        entity: Entity,
        field: InspectorLightField,
        delta: f32,
    },
    ToggleMeshCastsShadow(Entity),
    ToggleMeshVisible(Entity),
    SetTransformValue {
        entity: Entity,
        field: InspectorTransformField,
        value: f32,
    },
    SetCameraValue {
        entity: Entity,
        field: InspectorCameraField,
        value: f32,
    },
    SetLightValue {
        entity: Entity,
        field: InspectorLightField,
        value: f32,
    },
    SetLightColor {
        entity: Entity,
        color: [f32; 3],
    },
    ToggleLightColorPicker(Entity),
}

#[derive(Clone, Copy, Debug)]
pub enum InspectorPaneDragAction {
    Transform {
        entity: Entity,
        field: InspectorTransformField,
        sensitivity: f32,
    },
    Camera {
        entity: Entity,
        field: InspectorCameraField,
        sensitivity: f32,
    },
    Light {
        entity: Entity,
        field: InspectorLightField,
        sensitivity: f32,
    },
    LightColorSv {
        entity: Entity,
        surface_id: UiId,
    },
    LightColorHue {
        entity: Entity,
        surface_id: UiId,
    },
}

#[derive(Clone, Debug, Default)]
pub struct InspectorPaneFrame {
    pub actions: HashMap<UiId, InspectorPaneAction>,
    pub drag_actions: HashMap<UiId, InspectorPaneDragAction>,
    pub text_fields: HashMap<UiId, InspectorPaneTextField>,
}

pub fn build_inspector_pane(
    retained: &mut RetainedUi,
    root_id: UiId,
    viewport: UiRect,
    data: &InspectorPaneData,
) -> InspectorPaneFrame {
    let mut frame = InspectorPaneFrame::default();
    let pane_width = viewport.width.max(220.0);
    let content_width = (pane_width - 20.0).max(180.0);

    let background_id = root_id.child("background");
    retained.upsert(RetainedUiNode::new(
        root_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            clip: true,
            ..UiVisualStyle::default()
        }),
    ));
    retained.upsert(RetainedUiNode::new(
        background_id,
        UiWidget::Container,
        fill_style(UiVisualStyle {
            background: Some(UiColor::rgba(0.10, 0.12, 0.15, 0.84)),
            border_color: Some(UiColor::rgba(0.24, 0.30, 0.37, 0.9)),
            border_width: 1.0,
            corner_radius: 0.0,
            clip: true,
        }),
    ));

    let mut children = vec![background_id];
    let mut y = 8.0;

    let entity_label_id = root_id.child("entity-label");
    retained.upsert(RetainedUiNode::new(
        entity_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(format!("Entity: {}", data.entity_label)),
            style: UiTextStyle {
                color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
                font_size: 13.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, y, content_width, 22.0, UiVisualStyle::default()),
    ));
    children.push(entity_label_id);
    y += 24.0;

    let Some(entity) = data.entity else {
        let empty_id = root_id.child("empty");
        retained.upsert(RetainedUiNode::new(
            empty_id,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from("Select an entity to inspect components"),
                style: UiTextStyle {
                    color: UiColor::rgba(0.72, 0.76, 0.83, 1.0),
                    font_size: 12.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: true,
                },
            }),
            absolute_style(10.0, y + 4.0, content_width, 20.0, UiVisualStyle::default()),
        ));
        children.push(empty_id);
        retained.set_children(root_id, children);
        return frame;
    };

    let name_label_id = root_id.child("name-label");
    let name_field_id = root_id.child("name-field");
    retained.upsert(RetainedUiNode::new(
        name_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Name"),
            style: UiTextStyle {
                color: UiColor::rgba(0.90, 0.94, 1.0, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, y, 80.0, 22.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        name_field_id,
        UiWidget::TextField(UiTextField {
            text: UiTextValue::from(data.name_value.clone()),
            suffix: None,
            scroll_x: 0.0,
            style: UiTextStyle {
                color: UiColor::rgba(0.95, 0.97, 1.0, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
            enabled: true,
            focused: data.focused_field == Some(InspectorPaneTextField::Name),
            cursor: Some(
                data.text_cursors
                    .get(&InspectorPaneTextField::Name)
                    .copied()
                    .unwrap_or_else(|| data.name_value.chars().count())
                    .min(data.name_value.chars().count()),
            ),
            selection: None,
            show_caret: data.focused_field == Some(InspectorPaneTextField::Name),
            selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
            caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
        }),
        absolute_style(
            90.0,
            y,
            (content_width - 80.0).max(80.0),
            24.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.14, 0.17, 0.22, 0.94)),
                border_color: Some(UiColor::rgba(0.31, 0.38, 0.50, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    frame
        .text_fields
        .insert(name_field_id, InspectorPaneTextField::Name);
    children.push(name_label_id);
    children.push(name_field_id);
    y += 30.0;

    let id_row_id = root_id.child("entity-id-row");
    let id_label_id = id_row_id.child("label");
    retained.upsert(RetainedUiNode::new(
        id_row_id,
        UiWidget::Container,
        absolute_style(10.0, y, content_width, 20.0, UiVisualStyle::default()),
    ));
    let delete_btn_id = add_action_button(
        retained,
        &mut children,
        root_id,
        "entity-delete",
        "Delete",
        (content_width - 64.0).max(10.0),
        y,
        64.0,
    );
    retained.upsert(RetainedUiNode::new(
        id_label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(format!("ID: {}", entity.to_bits())),
            style: UiTextStyle {
                color: UiColor::rgba(0.76, 0.82, 0.90, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            10.0,
            y,
            (content_width - 74.0).max(20.0),
            20.0,
            UiVisualStyle::default(),
        ),
    ));
    frame
        .actions
        .insert(delete_btn_id, InspectorPaneAction::DeleteEntity(entity));
    children.push(id_row_id);
    children.push(id_label_id);
    y += 24.0;

    if data.has_transform {
        add_section_title(
            retained,
            &mut children,
            root_id,
            "transform-title",
            "Transform",
            y,
            content_width,
        );
        let remove_transform_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "transform-remove",
            "Remove",
            (content_width - 64.0).max(10.0),
            y - 1.0,
            64.0,
        );
        frame.actions.insert(
            remove_transform_btn,
            InspectorPaneAction::RemoveTransform(entity),
        );
        y += 20.0;

        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-position-x",
            "Position X",
            data.transform_position[0],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::PositionX),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::PositionX,
                sensitivity: 0.0125,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-position-y",
            "Position Y",
            data.transform_position[1],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::PositionY),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::PositionY,
                sensitivity: 0.0125,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-position-z",
            "Position Z",
            data.transform_position[2],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::PositionZ),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::PositionZ,
                sensitivity: 0.0125,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-rotation-x",
            "Rotation X",
            data.transform_rotation[0],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::RotationX),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::RotationX,
                sensitivity: 0.0625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-rotation-y",
            "Rotation Y",
            data.transform_rotation[1],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::RotationY),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::RotationY,
                sensitivity: 0.0625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-rotation-z",
            "Rotation Z",
            data.transform_rotation[2],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::RotationZ),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::RotationZ,
                sensitivity: 0.0625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-scale-x",
            "Scale X",
            data.transform_scale[0],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::ScaleX),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::ScaleX,
                sensitivity: 0.00625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-scale-y",
            "Scale Y",
            data.transform_scale[1],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::ScaleY),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::ScaleY,
                sensitivity: 0.00625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "transform-scale-z",
            "Scale Z",
            data.transform_scale[2],
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Transform(InspectorTransformField::ScaleZ),
            data.focused_field,
            InspectorPaneDragAction::Transform {
                entity,
                field: InspectorTransformField::ScaleZ,
                sensitivity: 0.00625,
            },
        );
        y += 4.0;
    }

    if data.has_camera {
        add_section_title(
            retained,
            &mut children,
            root_id,
            "camera-title",
            "Camera",
            y,
            content_width,
        );
        let remove_camera_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "camera-remove",
            "Remove",
            (content_width - 64.0).max(10.0),
            y - 1.0,
            64.0,
        );
        frame
            .actions
            .insert(remove_camera_btn, InspectorPaneAction::RemoveCamera(entity));
        y += 20.0;

        let active_label = if data.camera_active {
            "Game Camera (Active)"
        } else {
            "Set Game Camera"
        };
        let active_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "camera-active",
            active_label,
            10.0,
            y,
            200.0_f32.min(content_width),
        );
        frame
            .actions
            .insert(active_btn, InspectorPaneAction::SetActiveCamera(entity));
        y += 24.0;

        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "camera-fov",
            "FOV (deg)",
            data.camera_fov_deg,
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Camera(InspectorCameraField::FovDeg),
            data.focused_field,
            InspectorPaneDragAction::Camera {
                entity,
                field: InspectorCameraField::FovDeg,
                sensitivity: 0.0625,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "camera-aspect",
            "Aspect",
            data.camera_aspect_ratio,
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Camera(InspectorCameraField::Aspect),
            data.focused_field,
            InspectorPaneDragAction::Camera {
                entity,
                field: InspectorCameraField::Aspect,
                sensitivity: 0.0015,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "camera-near",
            "Near",
            data.camera_near,
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Camera(InspectorCameraField::Near),
            data.focused_field,
            InspectorPaneDragAction::Camera {
                entity,
                field: InspectorCameraField::Near,
                sensitivity: 0.0015,
            },
        );
        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "camera-far",
            "Far",
            data.camera_far,
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Camera(InspectorCameraField::Far),
            data.focused_field,
            InspectorPaneDragAction::Camera {
                entity,
                field: InspectorCameraField::Far,
                sensitivity: 0.125,
            },
        );
        y += 4.0;
    }

    if data.has_light {
        add_section_title(
            retained,
            &mut children,
            root_id,
            "light-title",
            "Light",
            y,
            content_width,
        );
        let remove_light_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "light-remove",
            "Remove",
            (content_width - 64.0).max(10.0),
            y - 1.0,
            64.0,
        );
        frame
            .actions
            .insert(remove_light_btn, InspectorPaneAction::RemoveLight(entity));
        y += 20.0;

        let type_labels = [
            (InspectorLightKind::Directional, "Directional"),
            (InspectorLightKind::Point, "Point"),
            (InspectorLightKind::Spot, "Spot"),
        ];
        let type_gap = 4.0;
        let type_width = ((content_width - type_gap * 2.0) / 3.0).max(52.0);
        for (index, (kind, label)) in type_labels.iter().enumerate() {
            let button_id = root_id.child("light-kind").child(index as u64);
            let selected = data.light_kind == *kind;
            retained.upsert(RetainedUiNode::new(
                button_id,
                UiWidget::Button(UiButton {
                    text: UiTextValue::from(*label),
                    variant: if selected {
                        UiButtonVariant::Primary
                    } else {
                        UiButtonVariant::Secondary
                    },
                    enabled: true,
                    style: UiTextStyle {
                        color: UiColor::rgba(0.93, 0.97, 1.0, 1.0),
                        font_size: 10.5,
                        align_h: UiTextAlign::Center,
                        align_v: UiTextAlign::Center,
                        wrap: false,
                    },
                }),
                absolute_style(
                    10.0 + index as f32 * (type_width + type_gap),
                    y,
                    type_width,
                    20.0,
                    UiVisualStyle {
                        background: Some(if selected {
                            UiColor::rgba(0.16, 0.42, 0.58, 0.95)
                        } else {
                            UiColor::rgba(0.17, 0.21, 0.27, 0.95)
                        }),
                        border_color: Some(if selected {
                            UiColor::rgba(0.36, 0.76, 0.95, 0.98)
                        } else {
                            UiColor::rgba(0.30, 0.39, 0.50, 0.90)
                        }),
                        border_width: 1.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            frame.actions.insert(
                button_id,
                InspectorPaneAction::SetLightType {
                    entity,
                    kind: *kind,
                },
            );
            children.push(button_id);
        }
        y += 24.0;

        y = add_light_color_picker(
            retained,
            &mut children,
            &mut frame,
            root_id,
            entity,
            y,
            content_width,
            data.light_color,
            data.light_color_hsv,
            data.light_color_picker_open,
        );

        y = add_step_row(
            retained,
            &mut children,
            &mut frame,
            root_id,
            "light-intensity",
            "Intensity",
            data.light_intensity,
            y,
            content_width,
            &data.text_cursors,
            InspectorPaneTextField::Light(InspectorLightField::Intensity),
            data.focused_field,
            InspectorPaneDragAction::Light {
                entity,
                field: InspectorLightField::Intensity,
                sensitivity: 0.0125,
            },
        );
        if data.light_kind == InspectorLightKind::Spot {
            y = add_step_row(
                retained,
                &mut children,
                &mut frame,
                root_id,
                "light-spot-angle",
                "Spot Angle",
                data.light_spot_angle_deg,
                y,
                content_width,
                &data.text_cursors,
                InspectorPaneTextField::Light(InspectorLightField::SpotAngleDeg),
                data.focused_field,
                InspectorPaneDragAction::Light {
                    entity,
                    field: InspectorLightField::SpotAngleDeg,
                    sensitivity: 0.0625,
                },
            );
        }
        y += 4.0;
    }

    if data.has_mesh_renderer {
        add_section_title(
            retained,
            &mut children,
            root_id,
            "mesh-title",
            "Mesh Renderer",
            y,
            content_width,
        );
        let remove_mesh_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "mesh-remove",
            "Remove",
            (content_width - 64.0).max(10.0),
            y - 1.0,
            64.0,
        );
        frame.actions.insert(
            remove_mesh_btn,
            InspectorPaneAction::RemoveMeshRenderer(entity),
        );
        y += 20.0;

        let mesh_line = root_id.child("mesh-ids");
        retained.upsert(RetainedUiNode::new(
            mesh_line,
            UiWidget::Label(UiLabel {
                text: UiTextValue::from(format!(
                    "mesh_id: {} | material_id: {}",
                    data.mesh_id, data.material_id
                )),
                style: UiTextStyle {
                    color: UiColor::rgba(0.82, 0.87, 0.95, 1.0),
                    font_size: 11.0,
                    align_h: UiTextAlign::Start,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(10.0, y, content_width, 20.0, UiVisualStyle::default()),
        ));
        children.push(mesh_line);
        y += 22.0;

        let casts_shadow_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "mesh-casts-shadow",
            if data.mesh_casts_shadow {
                "[x] Casts Shadow"
            } else {
                "[ ] Casts Shadow"
            },
            10.0,
            y,
            190.0_f32.min(content_width),
        );
        frame.actions.insert(
            casts_shadow_btn,
            InspectorPaneAction::ToggleMeshCastsShadow(entity),
        );

        let visible_btn = add_action_button(
            retained,
            &mut children,
            root_id,
            "mesh-visible",
            if data.mesh_visible {
                "[x] Visible"
            } else {
                "[ ] Visible"
            },
            (10.0 + (content_width * 0.52))
                .min(content_width - 96.0)
                .max(110.0),
            y,
            120.0,
        );
        frame
            .actions
            .insert(visible_btn, InspectorPaneAction::ToggleMeshVisible(entity));
        y += 24.0;
    }

    let add_component_btn = add_action_button(
        retained,
        &mut children,
        root_id,
        "add-component",
        "Add Component",
        10.0,
        y,
        128.0_f32.min(content_width),
    );
    frame.actions.insert(
        add_component_btn,
        InspectorPaneAction::OpenAddComponentMenu(entity),
    );

    retained.set_children(root_id, children);
    frame
}

fn add_section_title(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    root_id: UiId,
    key: &str,
    text: &str,
    y: f32,
    width: f32,
) {
    let id = root_id.child(key);
    retained.upsert(RetainedUiNode::new(
        id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(text.to_string()),
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.97, 1.0, 1.0),
                font_size: 12.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, y, width, 18.0, UiVisualStyle::default()),
    ));
    children.push(id);
}

fn add_step_row(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    frame: &mut InspectorPaneFrame,
    root_id: UiId,
    key: &str,
    label: &str,
    value: f32,
    y: f32,
    content_width: f32,
    text_cursors: &HashMap<InspectorPaneTextField, usize>,
    text_field: InspectorPaneTextField,
    focused_field: Option<InspectorPaneTextField>,
    drag_action: InspectorPaneDragAction,
) -> f32 {
    let label_width = (content_width * 0.42).clamp(108.0, 210.0);
    let value_width = (content_width - label_width - 10.0).max(82.0);
    let label_x = 10.0;
    let value_x = label_x + label_width + 6.0;
    let value_text = format!("{value:.4}");
    let label_id = root_id.child(key).child("label");
    let value_field_id = root_id.child(key).child("value-field");
    retained.upsert(RetainedUiNode::new(
        label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(label.to_string()),
            style: UiTextStyle {
                color: UiColor::rgba(0.82, 0.87, 0.95, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(label_x, y, label_width, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        value_field_id,
        UiWidget::TextField(UiTextField {
            text: UiTextValue::from(value_text.clone()),
            suffix: None,
            scroll_x: 0.0,
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.97, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
            enabled: true,
            focused: focused_field == Some(text_field),
            cursor: Some(
                text_cursors
                    .get(&text_field)
                    .copied()
                    .unwrap_or_else(|| value_text.chars().count())
                    .min(value_text.chars().count()),
            ),
            selection: None,
            show_caret: focused_field == Some(text_field),
            selection_color: UiColor::rgba(0.34, 0.52, 0.84, 0.46),
            caret_color: UiColor::rgba(0.96, 0.98, 1.0, 1.0),
        }),
        absolute_style(
            value_x,
            y,
            value_width,
            22.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.13, 0.17, 0.24, 0.90)),
                border_color: Some(UiColor::rgba(0.26, 0.34, 0.44, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    children.push(label_id);
    children.push(value_field_id);
    frame.drag_actions.insert(value_field_id, drag_action);
    frame.text_fields.insert(value_field_id, text_field);

    y + 24.0
}

#[allow(clippy::too_many_arguments)]
fn add_light_color_picker(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    frame: &mut InspectorPaneFrame,
    root_id: UiId,
    entity: Entity,
    y: f32,
    content_width: f32,
    light_color: [f32; 3],
    light_color_hsv: [f32; 3],
    picker_open: bool,
) -> f32 {
    let row_id = root_id.child("light-color-row");
    let label_id = row_id.child("label");
    let hex_id = row_id.child("hex");
    let swatch_bg_id = row_id.child("swatch-bg");
    let swatch_hit_id = row_id.child("swatch-hit");
    let swatch_w = content_width.clamp(52.0, 68.0) * 0.5;
    let swatch_w = swatch_w.clamp(52.0, 72.0);
    let swatch_h = 20.0;
    let swatch_x = (10.0 + content_width - swatch_w).max(10.0);
    let label_w = (swatch_x - 16.0).max(64.0);
    let hex = rgb_to_hex(light_color);
    retained.upsert(RetainedUiNode::new(
        row_id,
        UiWidget::Container,
        absolute_style(10.0, y, content_width, swatch_h, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        label_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Color"),
            style: UiTextStyle {
                color: UiColor::rgba(0.82, 0.87, 0.95, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(10.0, y, label_w, 20.0, UiVisualStyle::default()),
    ));
    retained.upsert(RetainedUiNode::new(
        hex_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from(hex),
            style: UiTextStyle {
                color: UiColor::rgba(0.72, 0.78, 0.88, 1.0),
                font_size: 10.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            54.0,
            y,
            (label_w - 44.0).max(1.0),
            20.0,
            UiVisualStyle::default(),
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        swatch_bg_id,
        UiWidget::Container,
        absolute_style(
            swatch_x,
            y,
            swatch_w,
            swatch_h,
            UiVisualStyle {
                background: Some(UiColor::rgba(
                    light_color[0].clamp(0.0, 1.0),
                    light_color[1].clamp(0.0, 1.0),
                    light_color[2].clamp(0.0, 1.0),
                    1.0,
                )),
                border_color: Some(if picker_open {
                    UiColor::rgba(0.62, 0.78, 0.98, 0.98)
                } else {
                    UiColor::rgba(0.30, 0.38, 0.49, 0.92)
                }),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        swatch_hit_id,
        UiWidget::HitBox,
        absolute_style(swatch_x, y, swatch_w, swatch_h, UiVisualStyle::default()),
    ));
    frame.actions.insert(
        swatch_hit_id,
        InspectorPaneAction::ToggleLightColorPicker(entity),
    );
    children.push(row_id);
    children.push(label_id);
    children.push(hex_id);
    children.push(swatch_bg_id);
    children.push(swatch_hit_id);

    let mut next_y = y + 24.0;
    if !picker_open {
        return next_y;
    }

    let popup_id = root_id.child("light-color-picker");
    let popup_x = 10.0;
    let popup_width = content_width.clamp(192.0, 252.0);
    let padding = 8.0;
    let header_h = 18.0;
    let sv_size = (popup_width - padding * 2.0).clamp(132.0, 200.0);
    let hue_h = 12.0;
    let presets_h = 20.0;
    let popup_height = padding + header_h + 6.0 + sv_size + 6.0 + hue_h + 6.0 + presets_h + padding;

    retained.upsert(RetainedUiNode::new(
        popup_id,
        UiWidget::Container,
        absolute_style(
            popup_x,
            next_y,
            popup_width,
            popup_height,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.09, 0.11, 0.14, 0.98)),
                border_color: Some(UiColor::rgba(0.28, 0.36, 0.46, 0.95)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));

    let mut popup_children = Vec::new();
    let title_id = popup_id.child("title");
    retained.upsert(RetainedUiNode::new(
        title_id,
        UiWidget::Label(UiLabel {
            text: UiTextValue::from("Light Color"),
            style: UiTextStyle {
                color: UiColor::rgba(0.92, 0.96, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Start,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            padding,
            padding,
            (popup_width - (padding * 3.0) - 20.0).max(1.0),
            header_h,
            UiVisualStyle::default(),
        ),
    ));
    popup_children.push(title_id);

    let close_id = popup_id.child("close");
    retained.upsert(RetainedUiNode::new(
        close_id,
        UiWidget::Button(UiButton {
            text: UiTextValue::from("x"),
            variant: UiButtonVariant::Ghost,
            enabled: true,
            style: UiTextStyle {
                color: UiColor::rgba(0.85, 0.90, 0.98, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            popup_width - padding - 18.0,
            padding,
            18.0,
            header_h,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.18, 0.22, 0.28, 0.95)),
                border_color: Some(UiColor::rgba(0.30, 0.38, 0.50, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    frame.actions.insert(
        close_id,
        InspectorPaneAction::ToggleLightColorPicker(entity),
    );
    popup_children.push(close_id);

    let sv_x = padding;
    let sv_y = padding + header_h + 6.0;
    let sv_root_id = popup_id.child("sv");
    retained.upsert(RetainedUiNode::new(
        sv_root_id,
        UiWidget::Container,
        absolute_style(
            sv_x,
            sv_y,
            sv_size,
            sv_size,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.0, 0.0, 0.0, 1.0)),
                border_color: Some(UiColor::rgba(0.28, 0.36, 0.48, 0.96)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    let mut sv_children = Vec::new();
    const SV_STEPS: usize = 20;
    let sv_cell_w = sv_size / SV_STEPS as f32;
    let sv_cell_h = sv_size / SV_STEPS as f32;
    for row in 0..SV_STEPS {
        for col in 0..SV_STEPS {
            let id = sv_root_id.child("c").child((row * SV_STEPS + col) as u64);
            let s = (col as f32 + 0.5) / SV_STEPS as f32;
            let v = 1.0 - (row as f32 + 0.5) / SV_STEPS as f32;
            let rgb = hsv_to_rgb(light_color_hsv[0], s, v);
            retained.upsert(RetainedUiNode::new(
                id,
                UiWidget::Container,
                absolute_style(
                    col as f32 * sv_cell_w,
                    row as f32 * sv_cell_h,
                    sv_cell_w + 0.75,
                    sv_cell_h + 0.75,
                    UiVisualStyle {
                        background: Some(UiColor::rgba(rgb[0], rgb[1], rgb[2], 1.0)),
                        border_color: None,
                        border_width: 0.0,
                        corner_radius: 0.0,
                        clip: false,
                    },
                ),
            ));
            sv_children.push(id);
        }
    }
    retained.set_children(sv_root_id, sv_children);
    popup_children.push(sv_root_id);

    let sv_hit_id = popup_id.child("sv-hit");
    retained.upsert(RetainedUiNode::new(
        sv_hit_id,
        UiWidget::HitBox,
        absolute_style(sv_x, sv_y, sv_size, sv_size, UiVisualStyle::default()),
    ));
    frame.drag_actions.insert(
        sv_hit_id,
        InspectorPaneDragAction::LightColorSv {
            entity,
            surface_id: sv_hit_id,
        },
    );

    let sv_indicator_x = sv_x + light_color_hsv[1].clamp(0.0, 1.0) * sv_size;
    let sv_indicator_y = sv_y + (1.0 - light_color_hsv[2].clamp(0.0, 1.0)) * sv_size;
    let sv_indicator_h = popup_id.child("sv-indicator-h");
    let sv_indicator_v = popup_id.child("sv-indicator-v");
    retained.upsert(RetainedUiNode::new(
        sv_indicator_h,
        UiWidget::Container,
        absolute_style(
            (sv_indicator_x - 5.0).max(sv_x),
            (sv_indicator_y - 0.5).max(sv_y),
            10.0,
            1.5,
            UiVisualStyle {
                background: Some(UiColor::rgba(1.0, 1.0, 1.0, 0.95)),
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    retained.upsert(RetainedUiNode::new(
        sv_indicator_v,
        UiWidget::Container,
        absolute_style(
            (sv_indicator_x - 0.5).max(sv_x),
            (sv_indicator_y - 5.0).max(sv_y),
            1.5,
            10.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(1.0, 1.0, 1.0, 0.95)),
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    popup_children.push(sv_indicator_h);
    popup_children.push(sv_indicator_v);
    popup_children.push(sv_hit_id);

    let hue_y = sv_y + sv_size + 6.0;
    let hue_id = popup_id.child("hue");
    retained.upsert(RetainedUiNode::new(
        hue_id,
        UiWidget::Container,
        absolute_style(
            sv_x,
            hue_y,
            sv_size,
            hue_h,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.0, 0.0, 0.0, 1.0)),
                border_color: Some(UiColor::rgba(0.28, 0.36, 0.48, 0.96)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: true,
            },
        ),
    ));
    let mut hue_children = Vec::new();
    const HUE_STEPS: usize = 48;
    let hue_cell_w = sv_size / HUE_STEPS as f32;
    for step in 0..HUE_STEPS {
        let id = hue_id.child("c").child(step as u64);
        let h = step as f32 / (HUE_STEPS - 1) as f32;
        let rgb = hsv_to_rgb(h, 1.0, 1.0);
        retained.upsert(RetainedUiNode::new(
            id,
            UiWidget::Container,
            absolute_style(
                step as f32 * hue_cell_w,
                0.0,
                hue_cell_w + 0.75,
                hue_h,
                UiVisualStyle {
                    background: Some(UiColor::rgba(rgb[0], rgb[1], rgb[2], 1.0)),
                    border_color: None,
                    border_width: 0.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        hue_children.push(id);
    }
    retained.set_children(hue_id, hue_children);
    popup_children.push(hue_id);

    let hue_hit_id = popup_id.child("hue-hit");
    retained.upsert(RetainedUiNode::new(
        hue_hit_id,
        UiWidget::HitBox,
        absolute_style(sv_x, hue_y, sv_size, hue_h, UiVisualStyle::default()),
    ));
    frame.drag_actions.insert(
        hue_hit_id,
        InspectorPaneDragAction::LightColorHue {
            entity,
            surface_id: hue_hit_id,
        },
    );

    let hue_indicator_id = popup_id.child("hue-indicator");
    let hue_indicator_x = sv_x + light_color_hsv[0].clamp(0.0, 1.0) * sv_size;
    retained.upsert(RetainedUiNode::new(
        hue_indicator_id,
        UiWidget::Container,
        absolute_style(
            (hue_indicator_x - 1.0).max(sv_x),
            hue_y - 1.0,
            2.0,
            hue_h + 2.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.97, 0.98, 1.0, 1.0)),
                border_color: None,
                border_width: 0.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    popup_children.push(hue_indicator_id);
    popup_children.push(hue_hit_id);

    let presets: [(&str, [f32; 3]); 6] = [
        ("W", [1.0, 1.0, 1.0]),
        ("R", [1.0, 0.0, 0.0]),
        ("G", [0.0, 1.0, 0.0]),
        ("B", [0.0, 0.0, 1.0]),
        ("Sun", [1.0, 0.95, 0.78]),
        ("Sky", [0.64, 0.80, 1.0]),
    ];
    let preset_y = hue_y + hue_h + 6.0;
    let preset_gap = 3.0;
    let preset_w =
        ((sv_size - preset_gap * (presets.len() as f32 - 1.0)) / presets.len() as f32).max(20.0);
    for (idx, (label, color)) in presets.iter().enumerate() {
        let button_id = popup_id.child("preset").child(idx as u64);
        retained.upsert(RetainedUiNode::new(
            button_id,
            UiWidget::Button(UiButton {
                text: UiTextValue::from(*label),
                variant: UiButtonVariant::Secondary,
                enabled: true,
                style: UiTextStyle {
                    color: UiColor::rgba(0.93, 0.97, 1.0, 1.0),
                    font_size: 10.0,
                    align_h: UiTextAlign::Center,
                    align_v: UiTextAlign::Center,
                    wrap: false,
                },
            }),
            absolute_style(
                sv_x + idx as f32 * (preset_w + preset_gap),
                preset_y,
                preset_w,
                presets_h,
                UiVisualStyle {
                    background: Some(UiColor::rgba(
                        (color[0] * 0.7).clamp(0.0, 1.0),
                        (color[1] * 0.7).clamp(0.0, 1.0),
                        (color[2] * 0.7).clamp(0.0, 1.0),
                        0.95,
                    )),
                    border_color: Some(UiColor::rgba(
                        color[0].clamp(0.0, 1.0),
                        color[1].clamp(0.0, 1.0),
                        color[2].clamp(0.0, 1.0),
                        0.95,
                    )),
                    border_width: 1.0,
                    corner_radius: 0.0,
                    clip: false,
                },
            ),
        ));
        frame.actions.insert(
            button_id,
            InspectorPaneAction::SetLightColor {
                entity,
                color: *color,
            },
        );
        popup_children.push(button_id);
    }

    retained.set_children(popup_id, popup_children);
    children.push(popup_id);
    next_y += popup_height + 6.0;
    next_y
}

fn add_action_button(
    retained: &mut RetainedUi,
    children: &mut Vec<UiId>,
    root_id: UiId,
    key: &str,
    text: &str,
    x: f32,
    y: f32,
    width: f32,
) -> UiId {
    let id = root_id.child("action").child(key);
    retained.upsert(RetainedUiNode::new(
        id,
        UiWidget::Button(UiButton {
            text: UiTextValue::from(text.to_string()),
            variant: UiButtonVariant::Secondary,
            enabled: true,
            style: UiTextStyle {
                color: UiColor::rgba(0.93, 0.96, 1.0, 1.0),
                font_size: 11.0,
                align_h: UiTextAlign::Center,
                align_v: UiTextAlign::Center,
                wrap: false,
            },
        }),
        absolute_style(
            x,
            y,
            width,
            20.0,
            UiVisualStyle {
                background: Some(UiColor::rgba(0.20, 0.27, 0.36, 0.90)),
                border_color: Some(UiColor::rgba(0.30, 0.40, 0.53, 0.90)),
                border_width: 1.0,
                corner_radius: 0.0,
                clip: false,
            },
        ),
    ));
    children.push(id);
    id
}

pub fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0].clamp(0.0, 1.0);
    let g = rgb[1].clamp(0.0, 1.0);
    let b = rgb[2].clamp(0.0, 1.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let mut h = 0.0;
    if delta > f32::EPSILON {
        if (max - r).abs() <= f32::EPSILON {
            h = ((g - b) / delta).rem_euclid(6.0) / 6.0;
        } else if (max - g).abs() <= f32::EPSILON {
            h = (((b - r) / delta) + 2.0) / 6.0;
        } else {
            h = (((r - g) / delta) + 4.0) / 6.0;
        }
    }
    let s = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    [h.rem_euclid(1.0), s.clamp(0.0, 1.0), max.clamp(0.0, 1.0)]
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    if s <= f32::EPSILON {
        return [v, v, v];
    }

    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i.rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

pub fn rgb_to_hex(rgb: [f32; 3]) -> String {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{r:02X}{g:02X}{b:02X}")
}

fn fill_style(visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .width(UiDimension::percent(1.0))
            .height(UiDimension::percent(1.0))
            .build(),
        visual,
    }
}

fn absolute_style(x: f32, y: f32, width: f32, height: f32, visual: UiVisualStyle) -> UiStyle {
    UiStyle {
        layout: UiLayoutBuilder::new()
            .position_type(helmer_ui::UiPositionType::Absolute)
            .left(UiDimension::points(x))
            .top(UiDimension::points(y))
            .width(UiDimension::points(width.max(0.0)))
            .height(UiDimension::points(height.max(0.0)))
            .build(),
        visual,
    }
}
