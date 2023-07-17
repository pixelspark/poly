<template>
  <div>
    <dl>
      <dt>Model:</dt>
      <dd>
        <select v-model="model">
          <option disabled>Select a model...</option>
          <option v-for="t in models" :value="t">{{ t }}</option>
        </select>
        <button @click="reload">Refresh</button>
      </dd>
    </dl>

    <dl>
      <dt>Text:</dt>
      <dd><textarea v-model="prompt"></textarea></dd>
    </dl>

    <dl>
      <dt></dt>
      <dd><button @click="run">Tokenize</button></dd>
    </dl>

    <dl>
      <dt></dt>
      <dd>
        <div v-if="result" class="tokenized">
          <span v-for="tok in result.tokens"> {{ tok.text }}</span>
        </div>
        <template v-if="result && wordCount > 0">
          {{ result.tokens.length }} tokens, {{ wordCount }} words,
          {{
            Math.round((result.tokens.length / wordCount) * 10) / 10
          }}
          tokens/word.
        </template>
      </dd>
    </dl>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, onMounted, Ref, computed } from "vue";

interface TokenizationResponse {
  tokens: { token: number; text: string }[];
}

const model = ref("");
const models = ref([]);
const get = inject("get") as (path: string) => any;
const post = inject("post") as (path: string, data: any) => any;
const prompt = ref("");
const result: Ref<TokenizationResponse | null> = ref(null);
const lastPrompt = ref("");

async function run() {
  const promptText = prompt.value;
  lastPrompt.value = prompt.value;
  result.value = await post(
    "v1/model/" + encodeURIComponent(model.value) + "/tokenization",
    { prompt: promptText }
  );
}

const wordCount = computed(() => {
  return lastPrompt.value.split(/[\ ]+/g).length;
});

async function reload() {
  models.value = (await get("v1/model")).models;
}

onMounted(() => reload());
</script>

<style scoped>
div.tokenized {
  padding: 5px;
}

div.tokenized span {
  border: solid 1px red;
  padding: 2px;
}
div.tokenized span:nth-child(2n) {
  background-color: rgb(230, 230, 230);
}
</style>
