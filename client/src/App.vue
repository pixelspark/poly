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

    <select v-model="screen">
      <option value="task">Task</option>
      <option value="embedding">Embedding</option>
      <option value="chat">Chat</option>
    </select>

    <Task v-if="screen === 'task'"></Task>
    <Embedding v-else-if="screen === 'embedding'"></Embedding>
    <Chat v-else-if="screen === 'chat'"></Chat>
  </div>
</template>

<script setup lang="ts">
import { onMounted, provide, ref, watch } from "vue";
import Task from "./components/Task.vue";
import Embedding from "./components/Embedding.vue";
import Chat from "./components/Chat.vue";

const screen = ref("chat");

const apiKey = ref("");
const base = ref(new URL("", document.location.toString()).toString());

watch(base, (nv) => {
  window.localStorage.setItem("llmd.base", nv.toString());
});

onMounted(() => {
  base.value = window.localStorage.getItem("llmd.base") || base.value;
});

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