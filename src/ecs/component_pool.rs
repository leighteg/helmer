use super::component::Component;

pub struct ComponentPool {
    components: Vec<Option<Box<dyn Component>>>,
}

impl ComponentPool {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    pub fn insert<T: Component>(&mut self, component: T) -> usize {
        self.components.push(Some(Box::new(component)));
        self.components.len() - 1
    }

    pub fn get(&self, id: usize) -> Option<&Box<dyn Component>> {
        self.components.get(id)?.as_ref()
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Box<dyn Component>> {
        self.components.get_mut(id)?.as_mut()
    }

    pub fn remove(&mut self, id: usize) {
        if let Some(slot) = self.components.get_mut(id) {
            *slot = None;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Box<dyn Component>> {
        self.components.iter().filter_map(|opt| opt.as_ref())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Component>> {
        self.components.iter_mut().filter_map(|opt| opt.as_mut())
    }
}
