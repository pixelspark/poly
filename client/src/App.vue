<template>
  <div>
    <dl>
      <dt>Base:</dt>
      <dd><input v-model="base" /></dd>
    </dl>
    <dl>
      <dt>API key:</dt>
      <dd><input v-model="apiKey" placeholder="Api key..." /></dd>
    </dl>

    <template v-if="authorized">
      <select v-model="screen">
        <option value="task">Task</option>
        <option value="embedding">Embedding</option>
        <option value="chat">Chat</option>
        <option value="memory">Memory</option>
        <option value="tokenization">Tokenization</option>
      </select>

      <Task v-if="screen === 'task'"></Task>
      <Embedding v-else-if="screen === 'embedding'"></Embedding>
      <Chat v-else-if="screen === 'chat'"></Chat>
      <Memory v-else-if="screen === 'memory'"></Memory>
      <Tokenization v-else-if="screen === 'tokenization'"></Tokenization>
    </template>
  </div>
</template>

<script setup lang="ts">
import { onMounted, provide, ref, watch } from "vue";
import Task from "./components/Task.vue";
import Embedding from "./components/Embedding.vue";
import Chat from "./components/Chat.vue";
import Memory from "./components/Memory.vue";
import Tokenization from "./components/Tokenization.vue";

const screen = ref("chat");

const apiKey = ref("");
const base = ref(new URL("", document.location.toString()).toString());
const authorized = ref(false);

watch(base, (nv) => {
  window.localStorage.setItem("llmd.base", nv.toString());
});

// TODO: fix this - when the func() completes in less than the timeout, it will happily run again
function debounce(func: () => Promise<void>, timeout = 300) {
  let running: Promise<void> | undefined;

  return async () => {
    if (running) {
      running = new Promise((resolve, _reject) => {
        running!.then(async () => {
          setTimeout(async () => {
            await func();
            running = undefined;
            resolve(undefined);
          }, timeout);
        });
      });
      await running;
    } else {
      running = func();
      await running;
      running = undefined;
    }
  };
}

const checkAuthorizedDebounced = debounce(checkAuthorized, 10000);

watch(apiKey, async (nv) => {
  window.localStorage.setItem("llmd.apiKey", nv.toString());
  await checkAuthorizedDebounced();
});

onMounted(async () => {
  base.value = window.localStorage.getItem("llmd.base") || base.value;
  apiKey.value = window.localStorage.getItem("llmd.apiKey") || apiKey.value;
  await checkAuthorizedDebounced();
});

async function checkAuthorized() {
  const url = new URL("/v1/task", base.value);
  if (apiKey.value.length > 0) {
    url.searchParams.append("api_key", apiKey.value);
  }
  const res = await fetch(url);
  console.log(res);
  authorized.value = res.status == 200;
}

async function get(path: string, body: any) {
  const url = new URL(path, base.value);
  for (const k in body) {
    if (Object.hasOwnProperty.call(body, k)) {
      url.searchParams.append(k, body[k]);
    }
  }
  if (apiKey.value.length > 0) {
    url.searchParams.append("api_key", apiKey.value);
  }
  console.log(url);
  const res = await fetch(url);
  return await res.json();
}

async function post(path: string, body: any) {
  const url = new URL(path, base.value);

  let authHeader: Record<string, string> = {};
  if (apiKey.value.length > 0) {
    authHeader["Authorization"] = "Bearer " + apiKey.value;
  }

  console.log(url);
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-type": "application/json",
      ...authHeader,
    },
    body: JSON.stringify(body),
  });
  return await res.json();
}

provide("get", get);
provide("post", post);
provide("base", base);
provide("apiKey", apiKey);
</script>
