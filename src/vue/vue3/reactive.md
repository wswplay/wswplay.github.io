---
title: Vue3.0响应式系统reactive源码分析
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

## getter

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

## setter

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
```
