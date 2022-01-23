---
title: 响应式系统 reactive
---

:::tip 感觉
Vue3.0 的 ReactiveEffect，就是 Vue2.0 的 Watch 嘛！
:::

## reactive(最新)

```js
function reactive(target) {
  // if trying to observe a readonly proxy, return the readonly version.
  if (target && target["__v_isReadonly" /* IS_READONLY */]) {
    return target;
  }
  return createReactiveObject(
    target,
    false,
    mutableHandlers,
    mutableCollectionHandlers,
    reactiveMap
  );
}
// createReactiveObject
function createReactiveObject(
  target,
  isReadonly,
  baseHandlers,
  collectionHandlers,
  proxyMap
) {
  debugger;
  if (!isObject(target)) {
    if (process.env.NODE_ENV !== "production") {
      console.warn(`value cannot be made reactive: ${String(target)}`);
    }
    return target;
  }
  // target is already a Proxy, return it.
  // exception: calling readonly() on a reactive object
  if (
    target["__v_raw" /* RAW */] &&
    !(isReadonly && target["__v_isReactive" /* IS_REACTIVE */])
  ) {
    return target;
  }
  // target already has corresponding Proxy
  const existingProxy = proxyMap.get(target);
  if (existingProxy) {
    return existingProxy;
  }
  // only a whitelist of value types can be observed.
  const targetType = getTargetType(target);
  if (targetType === 0 /* INVALID */) {
    return target;
  }
  const proxy = new Proxy(
    target,
    targetType === 2 /* COLLECTION */ ? collectionHandlers : baseHandlers
  );
  proxyMap.set(target, proxy);
  return proxy;
}
// baseHandlers
const mutableHandlers = {
  get,
  set,
  deleteProperty,
  has,
  ownKeys,
};
// get
const get = /*#__PURE__*/ createGetter();
```

## createGetter

```js
function createGetter(isReadonly = false, shallow = false) {
  return function get(target, key, receiver) {
    if (key === "__v_isReactive" /* IS_REACTIVE */) {
      return !isReadonly;
    } else if (key === "__v_isReadonly" /* IS_READONLY */) {
      return isReadonly;
    } else if (
      key === "__v_raw" /* RAW */ &&
      receiver ===
        (isReadonly
          ? shallow
            ? shallowReadonlyMap
            : readonlyMap
          : shallow
          ? shallowReactiveMap
          : reactiveMap
        ).get(target)
    ) {
      return target;
    }
    const targetIsArray = isArray(target);
    if (!isReadonly && targetIsArray && hasOwn(arrayInstrumentations, key)) {
      return Reflect.get(arrayInstrumentations, key, receiver);
    }
    const res = Reflect.get(target, key, receiver);
    if (isSymbol(key) ? builtInSymbols.has(key) : isNonTrackableKeys(key)) {
      return res;
    }
    if (!isReadonly) {
      track(target, "get" /* GET */, key);
    }
    if (shallow) {
      return res;
    }
    if (isRef(res)) {
      // ref unwrapping - does not apply for Array + integer key.
      const shouldUnwrap = !targetIsArray || !isIntegerKey(key);
      return shouldUnwrap ? res.value : res;
    }
    if (isObject(res)) {
      // Convert returned value into a proxy as well. we do the isObject check
      // here to avoid invalid value warning. Also need to lazy access readonly
      // and reactive here to avoid circular dependency.
      return isReadonly ? readonly(res) : reactive(res);
    }
    return res;
  };
}
```

## track

```js
function track(target, type, key) {
  if (!isTracking()) {
    return;
  }
  let depsMap = targetMap.get(target);
  if (!depsMap) {
    targetMap.set(target, (depsMap = new Map()));
  }
  let dep = depsMap.get(key);
  if (!dep) {
    depsMap.set(key, (dep = createDep()));
  }
  const eventInfo =
    process.env.NODE_ENV !== "production"
      ? { effect: activeEffect, target, type, key }
      : undefined;
  trackEffects(dep, eventInfo);
}
function trackEffects(dep, debuggerEventExtraInfo) {
  let shouldTrack = false;
  if (effectTrackDepth <= maxMarkerBits) {
    if (!newTracked(dep)) {
      dep.n |= trackOpBit; // set newly tracked
      shouldTrack = !wasTracked(dep);
    }
  } else {
    // Full cleanup mode.
    shouldTrack = !dep.has(activeEffect);
  }
  if (shouldTrack) {
    dep.add(activeEffect);
    activeEffect.deps.push(dep);
    if (process.env.NODE_ENV !== "production" && activeEffect.onTrack) {
      activeEffect.onTrack(
        Object.assign({ effect: activeEffect }, debuggerEventExtraInfo)
      );
    }
  }
}
```

## Reactive API(旧版？)

```js
function reactive(target) {
  // 如果尝试把一个 readonly proxy 变成响应式，直接返回这个 readonly proxy
  if (target && target.__v_isReadonly) {
    return target;
  }
  return createReactiveObject(
    target,
    false,
    mutableHandlers,
    mutableCollectionHandlers
  );
}
function createReactiveObject(
  target,
  isReadonly,
  baseHandlers,
  collectionHandlers
) {
  if (!isObject(target)) {
    // 目标必须是对象或数组类型
    if (process.env.NODE_ENV !== "production") {
      console.warn(`value cannot be made reactive: ${String(target)}`);
    }
    return target;
  }
  if (target.__v_raw && !(isReadonly && target.__v_isReactive)) {
    // target 已经是 Proxy 对象，直接返回
    // 有个例外，如果是 readonly 作用于一个响应式对象，则继续
    return target;
  }
  if (
    hasOwn(
      target,
      isReadonly ? "__v_readonly" /* readonly */ : "__v_reactive" /* reactive */
    )
  ) {
    // target 已经有对应的 Proxy 了
    return isReadonly ? target.__v_readonly : target.__v_reactive;
  }
  // 只有在白名单里的数据类型才能变成响应式
  if (!canObserve(target)) {
    return target;
  }
  // 利用 Proxy 创建响应式
  const observed = new Proxy(
    target,
    collectionTypes.has(target.constructor) ? collectionHandlers : baseHandlers
  );
  // 给原始数据打个标识，说明它已经变成响应式，并且有对应的 Proxy 了
  def(
    target,
    isReadonly ? "__v_readonly" /* readonly */ : "__v_reactive" /* reactive */,
    observed
  );
  return observed;
}
```

这么做是因为 Proxy 劫持的是对象本身，并不能劫持子对象的变化，这点和 Object.defineProperty API 一致。但是 Object.defineProperty 是在初始化阶段，即定义劫持对象的时候就已经递归执行了，而 Proxy 是在对象属性被访问的时候才递归执行下一步 reactive，这其实是一种延时定义子对象响应式的实现，在性能上会有较大的提升。
