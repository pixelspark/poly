<template>
  <div class="chat">
    <dl>
      <dt>Task:</dt>
      <dd>
        <select v-model="task" :disabled="!!socket">
          <option disabled>Select a task...</option>
          <option v-for="t in tasks" :value="t">{{ t }}</option>
        </select>
        <button @click="reload">Refresh</button>
      </dd>
    </dl>

    <dl v-if="task">
      <dt></dt>
      <dd>
        <button v-if="!socket" @click="connect">Connect</button>
        <button v-else @click="close">Disconnect</button>
      </dd>
    </dl>

    <dl v-if="socket">
      <dt>Chat:</dt>
      <dd>
        <ul class="chat">
          <li v-for="message in messages" :class="message.source">
            <div>
              <span>{{ message.text }}</span>
            </div>
          </li>
        </ul>
        <input
          v-model="userMessage"
          placeholder="Type a message..."
          @keyup.enter="send"
        />
      </dd>
    </dl>

    <dl>
      <dt></dt>
      <dd v-if="generatedTime > 0 && generatedTokens > 0">
        {{ generatedTokens }} tokens in
        {{ timeFormatter.format(generatedTime) }}s,
        {{ timeFormatter.format(generatedTokens / generatedTime) }} t/s
        <button
          @click="
            generatedTime = 0;
            generatedTokens = 0;
          "
        >
          Reset
        </button>
      </dd>
    </dl>
  </div>
</template>

<script setup lang="ts">
import { Ref, inject, onMounted, ref } from "vue";

const base = inject("base") as Ref<URL>;
const apiKey = inject("apiKey") as Ref<string>;
const get = inject("get") as (path: string) => any;

const socket = ref(null) as Ref<WebSocket | null>;
interface Message {
  source: string;
  text: string;
}
const messages = ref([]) as Ref<Message[]>;
const userMessage = ref("");
let lastServerMessage: Ref<Message | null> = ref(null);

const task = ref("");
const tasks = ref([]);
const generatingSince: Ref<null | number> = ref(null);
const generatedTokens = ref(0);
const generatedTime: Ref<number> = ref(0);

const timeFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

function send() {
  if (socket.value) {
    socket.value.send(userMessage.value);
    messages.value.push({ text: userMessage.value, source: "user" });
    userMessage.value = "";
    generatingSince.value = new Date().getTime();
  }
}

function close() {
  if (socket.value) {
    socket.value.close();
    socket.value = null;
    lastServerMessage.value = null;
  }
}

function connect() {
  close();
  messages.value = [];
  const url = new URL(
    "v1/task/" + encodeURIComponent(task.value) + "/chat",
    base.value
  );
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.append("api_key", apiKey.value);
  generatedTokens.value = 0;
  generatedTime.value = 0;
  generatingSince.value = null;

  socket.value = new WebSocket(url.toString());
  socket.value.onmessage = (me) => {
    generatedTokens.value++;
    const duration = (new Date().getTime() - generatingSince.value!) / 1000;
    generatedTime.value += duration;
    generatingSince.value = new Date().getTime();

    if (me.data === "") {
      lastServerMessage.value = null;
      return;
    }

    if (lastServerMessage.value === null) {
      lastServerMessage.value = { text: "", source: "server" };
      messages.value.push(lastServerMessage.value);
    }
    lastServerMessage.value!.text += me.data;
  };

  socket.value.onerror = () => {
    close();
  };

  socket.value.onclose = () => {
    socket.value = null;
  };
}

async function reload() {
  tasks.value = (await get("v1/task")).tasks;
}

onMounted(async () => {
  reload();
});
</script>

<style scoped>
div.chat {
  display: flex;
  flex-direction: column;
}

ul.chat {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

ul.chat li {
  list-style: none;
  margin-bottom: 3px;
  display: block;
  overflow: hidden;
}

ul.chat li.user {
  text-align: right;
}

ul.chat li div {
  padding: 5px;
  padding-bottom: 10px;
  padding-top: 10px;
  display: inline-block;
  border-radius: 15px;
  white-space: pre-wrap;
  line-height: 1.2em;
}

ul.chat li.user div {
  background: linear-gradient(rgba(0, 55, 100, 0.1), rgba(0, 55, 100, 0.2));
}

ul.chat li.server div {
  background: linear-gradient(rgba(236, 0, 0, 0.1), rgba(236, 0, 0, 0.2));
}
</style>
