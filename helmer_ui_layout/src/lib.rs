use glam::Vec2;
use hashbrown::HashMap;
use helmer_ui::UiId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl LayoutRect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(self) -> f32 {
        self.x + self.width
    }

    pub fn bottom(self) -> f32 {
        self.y + self.height
    }

    pub fn clamp_inside(self, bounds: Self) -> Self {
        let width = self.width.max(1.0).min(bounds.width.max(1.0));
        let height = self.height.max(1.0).min(bounds.height.max(1.0));
        let x = self
            .x
            .clamp(bounds.x, (bounds.right() - width).max(bounds.x));
        let y = self
            .y
            .clamp(bounds.y, (bounds.bottom() - height).max(bounds.y));
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl Default for LayoutRect {
    fn default() -> Self {
        Self::new(0.0, 0.0, 1.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WindowFrame {
    pub rect: LayoutRect,
    pub min_size: Vec2,
    pub max_size: Option<Vec2>,
    pub visible: bool,
    pub locked: bool,
}

impl Default for WindowFrame {
    fn default() -> Self {
        Self {
            rect: LayoutRect::new(0.0, 0.0, 320.0, 240.0),
            min_size: Vec2::new(120.0, 80.0),
            max_size: None,
            visible: true,
            locked: false,
        }
    }
}

impl WindowFrame {
    pub fn with_rect(mut self, rect: LayoutRect) -> Self {
        self.rect = rect;
        self
    }

    pub fn constrained_rect(&self) -> LayoutRect {
        let mut width = self.rect.width.max(self.min_size.x.max(1.0));
        let mut height = self.rect.height.max(self.min_size.y.max(1.0));
        if let Some(max) = self.max_size {
            width = width.min(max.x.max(self.min_size.x.max(1.0)));
            height = height.min(max.y.max(self.min_size.y.max(1.0)));
        }
        LayoutRect {
            width,
            height,
            ..self.rect
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct WindowLayoutState {
    windows: HashMap<UiId, WindowFrame>,
    z_order: Vec<UiId>,
}

#[derive(Clone, Debug, Default)]
pub struct WindowLayoutSnapshot {
    pub windows: HashMap<UiId, WindowFrame>,
    pub z_order: Vec<UiId>,
}

impl WindowLayoutState {
    pub fn clear(&mut self) {
        self.windows.clear();
        self.z_order.clear();
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    pub fn contains(&self, id: UiId) -> bool {
        self.windows.contains_key(&id)
    }

    pub fn ensure_window(&mut self, id: UiId, fallback: WindowFrame) -> &mut WindowFrame {
        if !self.z_order.contains(&id) {
            self.z_order.push(id);
        }
        self.windows.entry(id).or_insert(fallback)
    }

    pub fn add_window(&mut self, id: UiId, frame: WindowFrame) -> Option<WindowFrame> {
        if !self.z_order.contains(&id) {
            self.z_order.push(id);
        }
        self.windows.insert(id, frame)
    }

    pub fn remove_window(&mut self, id: UiId) -> Option<WindowFrame> {
        self.z_order.retain(|candidate| *candidate != id);
        self.windows.remove(&id)
    }

    pub fn window(&self, id: UiId) -> Option<&WindowFrame> {
        self.windows.get(&id)
    }

    pub fn window_mut(&mut self, id: UiId) -> Option<&mut WindowFrame> {
        self.windows.get_mut(&id)
    }

    pub fn set_rect(&mut self, id: UiId, rect: LayoutRect) {
        if let Some(frame) = self.windows.get_mut(&id) {
            frame.rect = rect;
        }
    }

    pub fn bring_to_front(&mut self, id: UiId) {
        if !self.windows.contains_key(&id) {
            return;
        }
        self.z_order.retain(|candidate| *candidate != id);
        self.z_order.push(id);
    }

    pub fn z_order(&self) -> &[UiId] {
        &self.z_order
    }

    pub fn ordered_visible_windows(&self) -> Vec<(UiId, &WindowFrame)> {
        self.z_order
            .iter()
            .filter_map(|id| {
                self.windows
                    .get(id)
                    .filter(|frame| frame.visible)
                    .map(|frame| (*id, frame))
            })
            .collect()
    }

    pub fn snapshot(&self) -> WindowLayoutSnapshot {
        WindowLayoutSnapshot {
            windows: self.windows.clone(),
            z_order: self.z_order.clone(),
        }
    }

    pub fn restore_snapshot(&mut self, snapshot: &WindowLayoutSnapshot) {
        self.windows = snapshot.windows.clone();
        self.z_order = snapshot
            .z_order
            .iter()
            .copied()
            .filter(|id| self.windows.contains_key(id))
            .collect();
    }

    pub fn normalize_to_bounds(&mut self, bounds: LayoutRect) {
        let ids: Vec<UiId> = self.windows.keys().copied().collect();
        for id in ids {
            if let Some(frame) = self.windows.get_mut(&id) {
                let constrained = frame.constrained_rect().clamp_inside(bounds);
                frame.rect = constrained;
            }
        }
    }

    pub fn rescale_to_bounds(&mut self, old_bounds: LayoutRect, new_bounds: LayoutRect) {
        if old_bounds.width <= 1.0
            || old_bounds.height <= 1.0
            || new_bounds.width <= 1.0
            || new_bounds.height <= 1.0
        {
            self.normalize_to_bounds(new_bounds);
            return;
        }

        let scale_x = new_bounds.width / old_bounds.width;
        let scale_y = new_bounds.height / old_bounds.height;
        let ids: Vec<UiId> = self.windows.keys().copied().collect();
        for id in ids {
            if let Some(frame) = self.windows.get_mut(&id) {
                let rel_x = frame.rect.x - old_bounds.x;
                let rel_y = frame.rect.y - old_bounds.y;
                frame.rect = LayoutRect::new(
                    new_bounds.x + rel_x * scale_x,
                    new_bounds.y + rel_y * scale_y,
                    frame.rect.width * scale_x,
                    frame.rect.height * scale_y,
                );
                frame.rect = frame.constrained_rect().clamp_inside(new_bounds);
            }
        }
    }

    pub fn tile_columns(&mut self, bounds: LayoutRect, columns: usize) {
        let columns = columns.max(1);
        let mut visible_ids: Vec<UiId> = self
            .z_order
            .iter()
            .copied()
            .filter(|id| self.windows.get(id).is_some_and(|window| window.visible))
            .collect();
        if visible_ids.is_empty() {
            return;
        }
        visible_ids.sort_by_key(|id| self.z_order.iter().position(|z| z == id).unwrap_or(0));

        let rows = visible_ids.len().div_ceil(columns);
        let cell_width = bounds.width / columns as f32;
        let cell_height = bounds.height / rows as f32;
        for (index, id) in visible_ids.into_iter().enumerate() {
            let row = index / columns;
            let col = index % columns;
            let x = bounds.x + col as f32 * cell_width;
            let y = bounds.y + row as f32 * cell_height;
            if let Some(window) = self.windows.get_mut(&id) {
                window.rect = LayoutRect::new(x, y, cell_width, cell_height);
            }
        }
    }

    pub fn tile_rows(&mut self, bounds: LayoutRect, rows: usize) {
        let rows = rows.max(1);
        let mut visible_ids: Vec<UiId> = self
            .z_order
            .iter()
            .copied()
            .filter(|id| self.windows.get(id).is_some_and(|window| window.visible))
            .collect();
        if visible_ids.is_empty() {
            return;
        }
        visible_ids.sort_by_key(|id| self.z_order.iter().position(|z| z == id).unwrap_or(0));

        let columns = visible_ids.len().div_ceil(rows);
        let cell_width = bounds.width / columns as f32;
        let cell_height = bounds.height / rows as f32;
        for (index, id) in visible_ids.into_iter().enumerate() {
            let row = index / columns;
            let col = index % columns;
            let x = bounds.x + col as f32 * cell_width;
            let y = bounds.y + row as f32 * cell_height;
            if let Some(window) = self.windows.get_mut(&id) {
                window.rect = LayoutRect::new(x, y, cell_width, cell_height);
            }
        }
    }

    pub fn cascade(&mut self, bounds: LayoutRect, offset: Vec2) {
        let mut cursor = Vec2::new(bounds.x, bounds.y);
        let visible_ids: Vec<UiId> = self
            .z_order
            .iter()
            .copied()
            .filter(|id| self.windows.get(id).is_some_and(|window| window.visible))
            .collect();
        for id in visible_ids {
            if let Some(window) = self.windows.get_mut(&id) {
                let mut rect = window.constrained_rect();
                rect.x = cursor.x;
                rect.y = cursor.y;
                window.rect = rect.clamp_inside(bounds);
                cursor += offset;
            }
        }
    }

    pub fn resize_with_reflow(
        &mut self,
        id: UiId,
        old_rect: LayoutRect,
        new_rect: LayoutRect,
        bounds: LayoutRect,
        tolerance: f32,
    ) {
        if !self.windows.contains_key(&id) {
            return;
        }
        let tol = tolerance.max(0.5);
        let delta_left = new_rect.x - old_rect.x;
        let delta_right = new_rect.right() - old_rect.right();
        let delta_top = new_rect.y - old_rect.y;
        let delta_bottom = new_rect.bottom() - old_rect.bottom();

        if let Some(target) = self.windows.get_mut(&id) {
            target.rect = new_rect;
            target.rect = target.constrained_rect().clamp_inside(bounds);
        }

        let mut visible_ids = Vec::new();
        for window_id in &self.z_order {
            if *window_id == id {
                continue;
            }
            if self
                .windows
                .get(window_id)
                .is_some_and(|window| window.visible)
            {
                visible_ids.push(*window_id);
            }
        }

        for window_id in visible_ids {
            let Some(window) = self.windows.get_mut(&window_id) else {
                continue;
            };
            let rect = window.rect;
            let mut next = rect;

            let overlap_y =
                ranges_overlap(old_rect.y, old_rect.bottom(), rect.y, rect.bottom(), tol);
            let overlap_x = ranges_overlap(old_rect.x, old_rect.right(), rect.x, rect.right(), tol);

            if overlap_y {
                if approx_eq(rect.x, old_rect.right(), tol) {
                    next.x += delta_right;
                    next.width -= delta_right;
                } else if approx_eq(rect.right(), old_rect.x, tol) {
                    next.width += delta_left;
                }
            }

            if overlap_x {
                if approx_eq(rect.y, old_rect.bottom(), tol) {
                    next.y += delta_bottom;
                    next.height -= delta_bottom;
                } else if approx_eq(rect.bottom(), old_rect.y, tol) {
                    next.height += delta_top;
                }
            }

            window.rect = next;
            window.rect = window.constrained_rect().clamp_inside(bounds);
        }

        self.normalize_to_bounds(bounds);
    }
}

fn approx_eq(lhs: f32, rhs: f32, tolerance: f32) -> bool {
    (lhs - rhs).abs() <= tolerance
}

fn ranges_overlap(a_min: f32, a_max: f32, b_min: f32, b_max: f32, tolerance: f32) -> bool {
    a_max >= b_min - tolerance && b_max >= a_min - tolerance
}
