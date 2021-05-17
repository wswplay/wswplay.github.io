---
title: 初始化
---
这里面有各种回调，有种JQuery的既视感。
```js
import Vuex from 'vuex'
Vue.use(Vuex) -> function install (_Vue) {
  Vue = _Vue;
  applyMixin(Vue) {
    if (version >= 2) Vue.mixin({ beforeCreate: vuexInit })
  } -> function vuexInit () {
    var options = this.$options;
    // store injection
    if (options.store) {
      this.$store = typeof options.store === 'function'
        ? options.store()
        : options.store;
    } else if (options.parent && options.parent.$store) {
      this.$store = options.parent.$store;
    }
  }
}

export default new Vuex.Store({
  state: {
  },
  mutations: {
  },
  actions: {
  },
  modules: {
  }
}) -> function Store (options) {
  // 初始化模块
  this._modules = new ModuleCollection(options) -> this.register()
  -> ModuleCollection.prototype.register() {
    var newModule = new Module(rawModule, runtime) {
      this.runtime = runtime;
      // Store some children item
      this._children = Object.create(null);
      // Store the origin module object which passed by programmer
      this._rawModule = rawModule;
      var rawState = rawModule.state;
      // Store the origin module's state
      this.state = (typeof rawState === 'function' ? rawState() : rawState) || {};
    }
    // 相互建立父子的树形结构关系
    if (path.length === 0) {
      this.root = newModule;
    } else {
      var parent = this.get(path.slice(0, -1));
      parent.addChild(path[path.length - 1], newModule);
    }
  }
  // 安装模块
  installModule(this, state, [], this._modules.root)
  -> var namespace = store._modules.getNamespace(path)
     var local = module.context = makeLocalContext(store, namespace, path)
    module.forEachMutation((mutation, key) => {
      const namespacedType = namespace + key
      registerMutation(store, namespacedType, mutation, local)
    })
    module.forEachAction((action, key) => {
      const type = action.root ? key : namespace + key
      const handler = action.handler || action
      registerAction(store, type, handler, local)
    })
    module.forEachGetter((getter, key) => {
      const namespacedType = namespace + key
      registerGetter(store, namespacedType, getter, local)
    })
    module.forEachChild((child, key) => {
      installModule(store, rootState, path.concat(key), child, hot)
    })
  // 初始化 store._vm
  resetStoreVM(this, state) {
    const computed = {}
    const wrappedGetters = store._wrappedGetters
    forEachValue(wrappedGetters, (fn, key) => {
      computed[key] = partial(fn, store)
      Object.defineProperty(store.getters, key, {
        get: () => store._vm[key],
        enumerable: true // for local getters
      })
    })
    store._vm = new Vue({
      data: {
        $$state: state
      },
      computed
    })
  }
  // 观察state
  get state () {
    return this._vm._data.$$state
  }
}
```
所以说对于 root module 的下一层 modules 来说，它们的 parent 就是 root module，那么他们就会被添加的 root module 的 _children 中。每个子模块通过路径找到它的父模块，然后通过父模块的 addChild 方法建立父子关系，递归执行这样的过程，最终就建立一颗完整的模块树。

## 参考
### Vue.use