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
      <dd><button @click="run">Calculate</button></dd>
    </dl>

    <table>
      <tr>
        <th></th>
        <th v-for="result in results" :title="result.prompt">
          <img :src="result.blobSource" :alt="result.prompt" />
          <br />{{ result.prompt.substring(0, 25) }}
        </th>
      </tr>

      <tr v-for="resultA in results">
        <td :title="resultA.prompt">
          <img
            :src="resultA.blobSource"
            :alt="resultA.prompt"
            style="float: left; margin: 3px"
          />{{ resultA.prompt.substring(0, 25) }}
        </td>
        <td
          v-for="resultB in results"
          :style="{
            backgroundColor: colorFor(
              cosineSimilarity(resultA.data, resultB.data)
            ),
            textAlign: 'right',
          }"
        >
          {{
            Math.round(cosineSimilarity(resultA.data, resultB.data) * 100) / 100
          }}
        </td>
      </tr>
    </table>

    <button @click="clear">Clear results</button>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, onMounted, Ref, onUnmounted } from "vue";

const model = ref("");
const models = ref([]);
const get = inject("get") as (path: string) => any;
const post = inject("post") as (path: string, data: any) => any;
const prompt = ref("");

interface EmbeddingResult {
  blobSource: string;
  prompt: string;
  data: number[];
}

const results = ref([]) as Ref<EmbeddingResult[]>;

async function reload() {
  models.value = (await get("v1/model")).models;
}

function colorFor(cs: number) {
  return `rgba(153,204,0,${Math.pow(cs, 2)})`;
}

function cosineSimilarity(dataA: number[], dataB: number[]) {
  var dotProduct = 0;
  var mA = 0;
  var mB = 0;

  for (let i = 0; i < dataA.length; i++) {
    dotProduct += dataA[i] * dataB[i];
    mA += dataA[i] * dataA[i];
    mB += dataB[i] * dataB[i];
  }

  mA = Math.sqrt(mA);
  mB = Math.sqrt(mB);
  const similarity = dotProduct / (mA * mB);
  return similarity;
}

async function run() {
  const promptText = prompt.value;
  const res = await post(
    "v1/model/" + encodeURIComponent(model.value) + "/embedding",
    { prompt: promptText }
  );

  const minValue = -1.0;
  const maxValue = 1.0;
  const data = res.embedding.map((v: number) => {
    const normalized =
      Math.max(minValue, Math.min(maxValue, v)) / (maxValue - minValue) + 0.5;
    return Math.pow(normalized, 2) * 255;
  });

  const canvas = document.createElement("CANVAS") as HTMLCanvasElement;
  const cellSize = 3;
  const cellsPerRow = Math.ceil(Math.sqrt(data.length)) / 3;
  const w = cellsPerRow * cellSize;
  canvas.width = w;
  canvas.height = w;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < data.length - 2; i += 3) {
    const cellIndex = Math.floor(i / 3);
    const color = `rgb(${data[i]}, ${data[i + 1]}, ${data[i + 2]})`;
    ctx.fillStyle = color;
    const x = cellIndex % cellsPerRow;
    const y = Math.floor(cellIndex / cellsPerRow);
    ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
  }

  canvas.toBlob((blob) => {
    if (blob) {
      const url = URL.createObjectURL(blob);
      results.value.push({ blobSource: url, prompt: promptText, data });
    }
  }, "image/png");
}

function clear() {
  results.value.forEach((result) => URL.revokeObjectURL(result.blobSource));
  results.value = [];
}

onUnmounted(() => {
  clear();
});

onMounted(() => reload());
</script>
