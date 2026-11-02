//! keel-kernel: everything is a plugin.
//!
//! A [`Context`] hosts services (typed singletons) and event listeners.
//! Plugins mount onto a context and register services/listeners through a
//! [`Scope`]. Every registration is an *effect* that is undone, in reverse
//! order, when the plugin unloads. There is no privileged core: the model
//! adapter, tools, session log and agent loop are all plugins.
//!
//! Inspired by Cordis (as used by DeepSeek Harness); written from scratch.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type Listener = Arc<dyn Fn(&dyn Any) + Send + Sync>;

#[derive(Default)]
struct Inner {
    services: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
    listeners: HashMap<TypeId, Vec<(u64, Listener)>>,
    next_id: u64,
}

/// Shared plugin host.
#[derive(Clone, Default)]
pub struct Context(Arc<Mutex<Inner>>);

/// A mountable unit of behaviour.
pub trait Plugin {
    fn name(&self) -> &str;
    fn apply(&self, scope: &mut Scope);
}

/// Handle to a mounted plugin; dropping or calling `unload` unwinds effects.
pub struct Mounted {
    effects: Vec<Box<dyn FnOnce() + Send>>,
}

impl Mounted {
    pub fn unload(mut self) {
        while let Some(undo) = self.effects.pop() {
            undo();
        }
    }
}

/// Registration surface handed to a plugin. Everything registered here is reversible.
pub struct Scope<'a> {
    ctx: &'a Context,
    effects: Vec<Box<dyn FnOnce() + Send>>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mount(&self, plugin: &dyn Plugin) -> Mounted {
        let mut scope = Scope { ctx: self, effects: Vec::new() };
        plugin.apply(&mut scope);
        Mounted { effects: scope.effects }
    }

    pub fn service<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        let g = self.0.lock().unwrap();
        g.services.get(&TypeId::of::<T>()).cloned()?.downcast::<T>().ok()
    }

    /// Fire an event to every listener registered for `E`.
    pub fn emit<E: Any>(&self, event: &E) {
        let ls: Vec<Listener> = {
            let g = self.0.lock().unwrap();
            g.listeners
                .get(&TypeId::of::<E>())
                .map(|v| v.iter().map(|(_, l)| l.clone()).collect())
                .unwrap_or_default()
        };
        for l in ls {
            l(event);
        }
    }
}

impl Scope<'_> {
    pub fn provide<T: Any + Send + Sync>(&mut self, svc: T) {
        let id = TypeId::of::<T>();
        self.ctx.0.lock().unwrap().services.insert(id, Arc::new(svc));
        let ctx = self.ctx.clone();
        self.effects.push(Box::new(move || {
            ctx.0.lock().unwrap().services.remove(&id);
        }));
    }

    pub fn on<E: Any, F: Fn(&E) + Send + Sync + 'static>(&mut self, f: F) {
        let tid = TypeId::of::<E>();
        let listener: Listener = Arc::new(move |e| {
            if let Some(e) = e.downcast_ref::<E>() {
                f(e)
            }
        });
        let lid = {
            let mut g = self.ctx.0.lock().unwrap();
            g.next_id += 1;
            let lid = g.next_id;
            g.listeners.entry(tid).or_default().push((lid, listener));
            lid
        };
        let ctx = self.ctx.clone();
        self.effects.push(Box::new(move || {
            if let Some(v) = ctx.0.lock().unwrap().listeners.get_mut(&tid) {
                v.retain(|(i, _)| *i != lid);
            }
        }));
    }

    pub fn context(&self) -> &Context {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Ping;
    struct Greeter(&'static str);
    struct Demo(Arc<AtomicUsize>);

    impl Plugin for Demo {
        fn name(&self) -> &str {
            "demo"
        }
        fn apply(&self, s: &mut Scope) {
            s.provide(Greeter("hi"));
            let c = self.0.clone();
            s.on::<Ping, _>(move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }
    }

    #[test]
    fn effects_unwind_on_unload() {
        let ctx = Context::new();
        let n = Arc::new(AtomicUsize::new(0));
        let m = ctx.mount(&Demo(n.clone()));
        assert_eq!(ctx.service::<Greeter>().unwrap().0, "hi");
        ctx.emit(&Ping);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        m.unload();
        ctx.emit(&Ping);
        assert_eq!(n.load(Ordering::SeqCst), 1);
        assert!(ctx.service::<Greeter>().is_none());
    }
}
