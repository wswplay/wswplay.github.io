---
title: 插槽slot
---

## 源码摘要

:::tip 简介
插槽的特点，其实就是在父组件中去编写子组件插槽部分的模板，然后在子组件渲染的时候，把这部分模板内容填充到子组件的插槽中。  
**对应到代码**：所以在父组件渲染阶段，子组件插槽部分的 DOM 是不能渲染的，需要通过某种方式保留下来，等到子组件渲染的时候再渲染。
:::

```js
function setupComponent(instance, isSSR = false) {
  // sth before
  initSlots(instance, children);
}
const initSlots = (instance, children) => {
  if (instance.vnode.shapeFlag & 32 /* SLOTS_CHILDREN */) {
    const type = children._;
    if (type) {
      instance.slots = toRaw(children);
      def(children, "_", type);
    } else {
      normalizeObjectSlots(children, (instance.slots = {}));
    }
  } else {
    instance.slots = {};
    if (children) {
      normalizeVNodeSlots(instance, children);
    }
  }
  def(instance.slots, InternalObjectKey, 1);
};
// VueLoader处理后和渲染
function renderSlot(slots, name, props = {}, fallback, noSlotted) {
  let slot = slots[name];
  openBlock();
  const validSlotContent = slot && ensureValidVNode(slot(props));
  const rendered = createBlock(
    Fragment,
    { key: props.key || `_${name}` },
    validSlotContent || (fallback ? fallback() : []),
    validSlotContent && slots._ === 1 /* STABLE */
      ? 64 /* STABLE_FRAGMENT */
      : -2 /* BAIL */
  );
  return rendered;
}
// 即renderSlot的入参fallback
function withCtx(fn, ctx = currentRenderingInstance, isNonScopedSlot) {
  if (!ctx) return fn;
  // already normalized
  if (fn._n) {
    return fn;
  }
  const renderFnWithContext = (...args) => {
    // sth
  };
  return renderFnWithContext;
}
const processFragment = () => {}
```

## 示例解析
