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

    <ul style="display: block; margin: 0; padding: 0">
      <li
        v-for="blob in results"
        style="display: inline-block; list-style: none; margin: 5px"
      >
        <img :src="blob" />
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { ref, inject, onMounted, Ref, onUnmounted } from "vue";

const model = ref("");
const models = ref([]);
const get = inject("get") as (path: string) => any;
const post = inject("post") as (path: string, data: any) => any;
const prompt = ref("");

const results = ref([]) as Ref<any[]>;

async function reload() {
  models.value = (await get("v1/model")).models;
}

async function run() {
  const res = await post(
    "v1/model/" + encodeURIComponent(model.value) + "/embedding",
    { prompt: prompt.value }
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
      results.value.push(url);
    }
  }, "image/png");
}

onUnmounted(() => {
  results.value.forEach((blobURL) => URL.revokeObjectURL(blobURL));
  results.value = [];
});

onMounted(() => reload());
</script>
