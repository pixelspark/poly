<template>
  <div class="task">
    <dl>
      <dt>Memory:</dt>
      <dd>
        <select v-model="memory">
          <option disabled>Select a memory...</option>
          <option v-for="t in memories" :value="t">{{ t }}</option>
        </select>
        <button @click="reload">Refresh</button>
      </dd>
    </dl>

    <template v-if="memory">
      <dl>
        <dt>Input:</dt>
        <dd>
          <input
            v-model="prompt"
            placholder="Some text..."
            @keyup.enter="retrieve"
          />
        </dd>
      </dl>

      <dl>
        <dt></dt>
        <dd>
          <button :disabled="!memory || !prompt || storing" @click="retrieve">
            Retrieve
          </button>
          <button :disabled="!memory || !prompt || storing" @click="store">
            Store
          </button>
          <button :disabled="!memory || storing" @click="upload">
            Upload file...
          </button>
        </dd>
      </dl>

      <dl>
        <dt></dt>
        <dd>
          <ul>
            <li v-for="m in response">{{ m }}</li>
          </ul>
        </dd>
      </dl>
    </template>
  </div>
</template>

<script setup lang="ts">
import { Ref, inject, onMounted, ref } from "vue";

const base = inject("base") as Ref<URL>;
const apiKey = inject("apiKey") as Ref<string>;
const get = inject("get") as (path: string, params?: any) => any;

const prompt = ref("");
const memory = ref("");
const memories = ref([]);
const response = ref([]);
const storing = ref(false);

async function reload() {
  memories.value = (await get("v1/memory")).memories;
}

async function storeFile(mime: string, body: string | File) {
  if (memory.value && prompt.value && !storing.value) {
    storing.value = true;
    const headers: Record<string, any> = {
      "Content-type": mime,
    };
    if (apiKey.value.length > 0) {
      headers["Authorization"] = "Bearer " + apiKey.value;
    }
    const url = new URL(
      "v1/memory/" + encodeURIComponent(memory.value),
      base.value
    );

    await fetch(url, {
      method: "POST",
      headers,
      body,
    });
    storing.value = false;
  }
}

async function store() {
  return await storeFile("text/plain", prompt.value);
}

async function retrieve() {
  if (memory.value && prompt.value) {
    const res = await get("v1/memory/" + encodeURIComponent(memory.value), {
      prompt: prompt.value,
      n: 3,
    });
    response.value = res.memories;
  }
}

async function selectFile(): Promise<File> {
  return await new Promise((resolve, reject) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf,.txt,.docx,text/plain";

    input.addEventListener("change", (evt: any) => {
      if (evt.target.files.length > 0) {
        const file = evt.target.files[0];
        console.log({ file });
        resolve(file);
      } else {
        reject(new Error("no file selected"));
      }
    });
    input.click();
  });
}

async function upload() {
  const file = await selectFile();
  if (file) {
    await storeFile(file.type, file);
  }
}

onMounted(async () => {
  reload();
});
</script>

<style scoped>
div.memory {
  display: flex;
  flex-direction: column;
}
</style>
