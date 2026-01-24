use serde::{Deserialize, Serialize};

use bevy_ecs::prelude::Component;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicComponent {
    pub name: String,
    pub fields: Vec<DynamicField>,
}

impl DynamicComponent {
    pub fn new(name: String) -> Self {
        Self {
            name,
            fields: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicField {
    pub name: String,
    pub value: DynamicValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicValue {
    Bool(bool),
    Float(f32),
    Int(i32),
    Vec3([f32; 3]),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicValueKind {
    Bool,
    Float,
    Int,
    Vec3,
    String,
}

impl DynamicValueKind {
    pub fn default_value(self) -> DynamicValue {
        match self {
            DynamicValueKind::Bool => DynamicValue::Bool(false),
            DynamicValueKind::Float => DynamicValue::Float(0.0),
            DynamicValueKind::Int => DynamicValue::Int(0),
            DynamicValueKind::Vec3 => DynamicValue::Vec3([0.0, 0.0, 0.0]),
            DynamicValueKind::String => DynamicValue::String(String::new()),
        }
    }
}

#[derive(Component, Debug, Clone, Default)]
pub struct DynamicComponents {
    pub components: Vec<DynamicComponent>,
}
