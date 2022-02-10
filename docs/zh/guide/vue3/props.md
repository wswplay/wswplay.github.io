---
title: Props
---

## 源码摘要

```js
function setupComponent(instance, isSSR = false) {
  // sth
  initProps(instance, props, isStateful, isSSR);
  // more sth
}
function initProps(...args) {
  const props = {};
  const attrs = {};
  def(attrs, InternalObjectKey, 1);
  instance.propsDefaults = Object.create(null);
  // 求值
  setFullProps(instance, rawProps, props, attrs);
  // ensure all declared prop keys are present
  for (const key in instance.propsOptions[0]) {
    if (!(key in props)) {
      props[key] = undefined;
    }
  }
  // 验证
  if (process.env.NODE_ENV !== "production") {
    validateProps(rawProps || {}, props, instance);
  }
  // 挂载到实例
  if (isStateful) {
    // stateful
    instance.props = isSSR ? props : shallowReactive(props);
  } else {
    if (!instance.type.props) {
      // functional w/ optional props, props === attrs
      instance.props = attrs;
    } else {
      // functional w/ declared props
      instance.props = props;
    }
  }
  instance.attrs = attrs;
}
```
## 示例解析
```vue
<template>
  <div @click="changeMsg">
    <HelloWorld :msg="miao"></HelloWorld>
  </div>
</template>

<script>
import HelloWorld from "./components/HelloWorld.vue";

export default {
  name: "App",
  components: { HelloWorld },
  data() {
    return {
      miao: "喵~",
    };
  },
  methods: {
    changeMsg() {
      this.miao = "不可能永远在黑暗里独舞";
    },
  },
};
</script>
```
:::tip
注意```shouldUpdateComponent```
:::
