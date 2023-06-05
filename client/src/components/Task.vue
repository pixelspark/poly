<template>
  <div class="task">
    <dl>
      <dt>Task:</dt>
      <dd>
        <select v-model="task" :disabled="!!stream">
          <option disabled>Select a task...</option>
          <option v-for="t in tasks" :value="t">{{ t }}</option>
        </select>
        <button @click="reload">Refresh</button>
      </dd>
    </dl>

    <dl>
      <dt>Prompt:</dt>
      <dd>
        <input
          v-model="prompt"
          :disabled="!!stream"
          placholder="Prompt..."
          @keyup.enter="run"
        />
      </dd>
    </dl>

    <dl>
      <dt></dt>
      <dd>
        <button v-if="stream" @click="stop">Stop</button>
        <button v-if="!stream" :disabled="!task || !prompt" @click="run">
          Run
        </button>
      </dd>
    </dl>

    <dl>
      <dt></dt>
      <dd>
        <textarea readonly style="min-height: 200px">{{ response }}</textarea>
      </dd>
    </dl>
  </div>
</template>

<script setup lang="ts">
import { Ref, inject, onMounted, ref } from "vue";

const base = inject("base") as Ref<URL>;
const apiKey = inject("apiKey") as Ref<string>;
const get = inject("get") as (path: string) => any;

const prompt = ref("");
const task = ref("");
const tasks = ref([]);
const response = ref("");

let stream: Ref<EventSource | null> = ref(null);

function stop() {
  if (stream.value) {
    stream.value.close();
    stream.value = null;
  }
}

async function reload() {
  tasks.value = (await get("v1/task")).tasks;
}

function run() {
  stop();
  response.value = "";
  const url = new URL(
    "v1/task/" + encodeURIComponent(task.value) + "/live",
    base.value
  );
  url.searchParams.append("prompt", prompt.value);
  url.searchParams.append("api_key", apiKey.value);
  stream.value = new EventSource(url);

  stream.value.onerror = function () {
    stop();
  };

  stream.value.onmessage = function (e) {
    response.value += e.data;
  };
}

onMounted(async () => {
  reload();
});
</script>

<style scoped>
div.task {
  display: flex;
  flex-direction: column;
}
</style>
