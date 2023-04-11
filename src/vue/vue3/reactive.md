---
title: Vue3.0响应式系统reactive源码分析
outline: deep
---

# Reactive：响应式系统

## reactive 函数

```ts {8}
export const reactiveMap = new WeakMap<Target, any>()
export function reactive(target: object) {
  if (isReadonly(target)) return target
  return createReactiveObject(target, false, mutableHandlers, collectionHandlers, reactiveMap) {
    // createReactiveObject(target, false, baseHandlers, collectionHandlers, proxyMap)
    const existingProxy = proxyMap.get(target)
    if (existingProxy) { return existingProxy }
    const proxy = new Proxy(
      target,
      argetType === TargetType.COLLECTION ? collectionHandlers : baseHandlers
    )
    proxyMap.set(target, proxy)
    return proxy
  }
}
export const mutableHandlers: ProxyHandler<object> = {
  get,
  set,
  deleteProperty,
  has,
  ownKeys,
};
export const collectionHandlers: ProxyHandler<CollectionTypes> = {
  get: /*#__PURE__*/ createInstrumentationGetter(false, false),
};
```

### getter

```ts
// getter
const get = /*#__PURE__*/ createGetter();
function createGetter(isReadonly = false, shallow = false) {
  return function get(target: Target, key: string | symbol, receiver: object) {
    if (xxx) return target;
    // 如果数组
    const targetIsArray = isArray(target);
    if (!isReadonly) {
      if (targetIsArray && hasOwn(arrayInstrumentations, key)) {
        return Reflect.get(arrayInstrumentations, key, receiver);
      }
    }
    // 声明结果值
    const res = Reflect.get(target, key, receiver);
    // 重要了注意了！！！建立追踪
    if (!isReadonly) track(target, TrackOpTypes.GET, key);
    // 如果浅响应式
    if (shallow) return res;
    // 如果ref值
    if (isRef(res)) return targetIsArray && isIntegerKey(key) ? res : res.value;
    // 如果对象，则递归执行相关操作
    if (isObject(res)) return isReadonly ? readonly(res) : reactive(res);
    // 返回结果值
    return res;
  };
}
export function track(target: object, type: TrackOpTypes, key: unknown) {
  if (shouldTrack && activeEffect) {
    let depsMap = targetMap.get(target);
    if (!depsMap) {
      targetMap.set(target, (depsMap = new Map()));
    }
    let dep = depsMap.get(key);
    if (!dep) {
      depsMap.set(key, (dep = createDep()));
    }
    const eventInfo = __DEV__
      ? { effect: activeEffect, target, type, key }
      : undefined;
    trackEffects(dep, eventInfo);
  }
}
export function trackEffects(
  dep: Dep,
  debuggerEventExtraInfo?: DebuggerEventExtraInfo
) {
  let shouldTrack = false;
  if (effectTrackDepth <= maxMarkerBits) {
    if (!newTracked(dep)) {
      dep.n |= trackOpBit; // set newly tracked
      shouldTrack = !wasTracked(dep);
    }
  } else {
    // Full cleanup mode.
    shouldTrack = !dep.has(activeEffect!);
  }
  if (shouldTrack) {
    dep.add(activeEffect!);
    activeEffect!.deps.push(dep);
  }
}
```

### setter

```ts
// setter
const set = /*#__PURE__*/ createSetter();
function createSetter(shallow = false) {
  return function set(target, key, value, receiver) {
    let oldValue = (target as any)[key];
    if (isReadonly(oldValue) && isRef(oldValue) && !isRef(value)) {
      return false;
    }
    const result = Reflect.set(target, key, value, receiver);
    if (target === toRaw(receiver)) {
      if (!hadKey) {
        trigger(target, TriggerOpTypes.ADD, key, value);
      } else if (hasChanged(value, oldValue)) {
        trigger(target, TriggerOpTypes.SET, key, value, oldValue);
      }
    }
    return result;
  };
}
export function trigger(target, type, key, newValue, oldValue, oldTarget) {
  const depsMap = targetMap.get(target);
  if (!depsMap) {
    return;
  }
  let deps: (Dep | undefined)[] = [];
  deps.push(depsMap.get(key));
  switch (type) {
    case TriggerOpTypes.ADD:
      if (!isArray(target)) {
        deps.push(depsMap.get(ITERATE_KEY));
        if (isMap(target)) {
          deps.push(depsMap.get(MAP_KEY_ITERATE_KEY));
        }
      } else if (isIntegerKey(key)) {
        // new index added to array -> length changes
        deps.push(depsMap.get("length"));
      }
      break;
    case TriggerOpTypes.DELETE:
      if (!isArray(target)) {
        deps.push(depsMap.get(ITERATE_KEY));
        if (isMap(target)) {
          deps.push(depsMap.get(MAP_KEY_ITERATE_KEY));
        }
      }
      break;
    case TriggerOpTypes.SET:
      if (isMap(target)) {
        deps.push(depsMap.get(ITERATE_KEY));
      }
      break;
  }
  if (deps.length === 1) {
    triggerEffects(deps[0]);
  } else {
    const effects: ReactiveEffect[] = [];
    for (const dep of deps) {
      if (dep) {
        effects.push(...dep);
      }
    }
    triggerEffects(createDep(effects));
  }
}
export function triggerEffects(
  dep: Dep | ReactiveEffect[],
  debuggerEventExtraInfo?: DebuggerEventExtraInfo
) {
  // spread into array for stabilization
  const effects = isArray(dep) ? dep : [...dep];
  for (const effect of effects) {
    if (effect.computed) {
      triggerEffect(effect, debuggerEventExtraInfo);
    }
  }
  for (const effect of effects) {
    if (!effect.computed) {
      triggerEffect(effect, debuggerEventExtraInfo);
    }
  }
}
function triggerEffect(
  effect: ReactiveEffect,
  debuggerEventExtraInfo?: DebuggerEventExtraInfo
) {
  if (effect !== activeEffect || effect.allowRecurse) {
    if (__DEV__ && effect.onTrigger) {
      effect.onTrigger(extend({ effect }, debuggerEventExtraInfo));
    }
    if (effect.scheduler) {
      effect.scheduler();
    } else {
      effect.run();
    }
  }
}
```

## computed 计算属性

```ts
export function computed<T>(
  getterOrOptions: ComputedGetter<T> | WritableComputedOptions<T>,
  debugOptions?: DebuggerOptions,
  isSSR = false
) {
  let getter: ComputedGetter<T>;
  let setter: ComputedSetter<T>;
  const onlyGetter = isFunction(getterOrOptions);
  if (onlyGetter) {
    getter = getterOrOptions;
    setter = __DEV__
      ? () => {
          console.warn("Write operation failed: computed value is readonly");
        }
      : NOOP;
  } else {
    getter = getterOrOptions.get;
    setter = getterOrOptions.set;
  }
  const cRef = new ComputedRefImpl(
    getter,
    setter,
    onlyGetter || !setter,
    isSSR
  );
  return cRef as any;
}
export class ComputedRefImpl<T> {
  public dep?: Dep = undefined;
  public readonly effect: ReactiveEffect<T>;
  public readonly __v_isRef = true;
  public _dirty = true;
  constructor(getter, _setter, isReadonly, isSSR) {
    this.effect = new ReactiveEffect(getter, () => {
      if (!this._dirty) {
        this._dirty = true;
        triggerRefValue(this);
      }
    });
    this.effect.computed = this;
    this.effect.active = this._cacheable = !isSSR;
  }
  get value() {
    const self = toRaw(this);
    trackRefValue(self);
    if (self._dirty || !self._cacheable) {
      self._dirty = false;
      self._value = self.effect.run()!;
    }
    return self._value;
  }

  set value(newValue: T) {
    this._setter(newValue);
  }
}
export function triggerRefValue(ref: RefBase<any>, newVal?: any) {
  ref = toRaw(ref);
  const dep = ref.dep;
  if (dep) triggerEffects(dep);
}
export function trackRefValue(ref: RefBase<any>) {
  if (shouldTrack && activeEffect) {
    ref = toRaw(ref);
    trackEffects(ref.dep || (ref.dep = createDep()));
  }
}
```

## shallowReactive 浅响应式

```ts
export function shallowReactive<T extends object>(
  target: T
): ShallowReactive<T> {
  return createReactiveObject(
    target,
    false,
    shallowReactiveHandlers,
    shallowCollectionHandlers,
    shallowReactiveMap
  );
}
const shallowGet = /*#__PURE__*/ createGetter(false, true);
const shallowSet = /*#__PURE__*/ createSetter(true);
export const shallowReactiveHandlers = /*#__PURE__*/ extend(
  {},
  mutableHandlers,
  {
    get: shallowGet,
    set: shallowSet,
  }
);
```

## readonly 只读属性

```ts
export function readonly<T extends object>(
  target: T
): DeepReadonly<UnwrapNestedRefs<T>> {
  return createReactiveObject(
    target,
    true,
    readonlyHandlers,
    readonlyCollectionHandlers,
    readonlyMap
  );
}
const readonlyGet = /*#__PURE__*/ createGetter(true);
export const readonlyHandlers: ProxyHandler<object> = {
  get: readonlyGet,
  set(target, key) {
    if (__DEV__) {
      warn(
        `Set operation on key "${String(key)}" failed: target is readonly.`,
        target
      );
    }
    return true;
  },
  deleteProperty(target, key) {
    if (__DEV__) {
      warn(
        `Delete operation on key "${String(key)}" failed: target is readonly.`,
        target
      );
    }
    return true;
  },
};
```

## ref 基础类型值响应式

将`目标值`存储在`RefImpl`类型对象的`value`key 上，用`get/set`(取/设)`value`key 值。

```ts
export function ref(value?: unknown) {
  return createRef(value, false);
}
function createRef(rawValue: unknown, shallow: boolean) {
  if (isRef(rawValue)) {
    return rawValue;
  }
  return new RefImpl(rawValue, shallow);
}
class RefImpl<T> {
  private _value: T;
  private _rawValue: T;
  public dep?: Dep = undefined;
  public readonly __v_isRef = true;
  constructor(value: T, public readonly __v_isShallow: boolean) {
    this._rawValue = __v_isShallow ? value : toRaw(value);
    this._value = __v_isShallow ? value : toReactive(value);
  }
  get value() {
    trackRefValue(this);
    return this._value;
  }
  set value(newVal) {
    const useDirectValue =
      this.__v_isShallow || isShallow(newVal) || isReadonly(newVal);
    newVal = useDirectValue ? newVal : toRaw(newVal);
    if (hasChanged(newVal, this._rawValue)) {
      this._rawValue = newVal;
      this._value = useDirectValue ? newVal : toReactive(newVal);
      triggerRefValue(this, newVal);
    }
  }
}
```

## 辅助信息集锦

```ts
export const enum TrackOpTypes {
  GET = "get",
  HAS = "has",
  ITERATE = "iterate",
}
export const enum TriggerOpTypes {
  SET = "set",
  ADD = "add",
  DELETE = "delete",
  CLEAR = "clear",
}
export const createDep = (effects?: ReactiveEffect[]): Dep => {
  const dep = new Set<ReactiveEffect>(effects) as Dep;
  dep.w = 0;
  dep.n = 0;
  return dep;
};
export let activeEffect: ReactiveEffect | undefined;
export class ReactiveEffect<T = any> {
  active = true;
  deps: Dep[] = [];
  constructor(
    public fn: () => T,
    public scheduler: EffectScheduler | null = null,
    scope?: EffectScope
  ) {
    recordEffectScope(this, scope);
  }
  run() {
    if (!this.active) return this.fn();
    let parent: ReactiveEffect | undefined = activeEffect;
    try {
      this.parent = activeEffect;
      activeEffect = this;
      shouldTrack = true;
      return this.fn();
    } finally {
    }
  }
}
```
