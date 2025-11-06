<script lang="ts">
  import { sendToLlama } from "$lib/services/chat";

  let messages: { sender: "user" | "ai"; text: string }[] = [];
  let userInput = "";
  let started = false;

  async function sendMessage() {
    if (!userInput.trim()) return;

    if (!started) started = true;

    // Add user's message
    messages = [...messages, { sender: "user", text: userInput }];

    let inputCopy = userInput;
    userInput = "";

    // ✅ Send to LLaMA
    // ✅ Call backend RAG + llama.cpp
    const aiReply = await sendToLlama(messages, inputCopy, "session-1");


    // Push AI reply
    messages = [...messages, { sender: "ai", text: aiReply }];
}
</script>


<!-- Welcome Screen -->
{#if !started}
<div class="flex flex-col items-center justify-center h-full text-center space-y-4">
  <h1 class="text-5xl font-bold">TanishGPT</h1>
  <p class="text-gray-600 dark:text-gray-300 max-w-md">
    Your private personal AI assistant with memory — running fully on your device.
  </p>

  <button
    class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
    on:click={() => (started = true)}>
    Start chatting
  </button>
</div>

{:else}

<!-- Chat UI -->
<div class="flex flex-col h-full">

  <!-- Messages -->
  <div class="flex-1 space-y-4 overflow-y-auto p-2">
    {#each messages as msg}
      <div class={`p-3 rounded-lg max-w-xl
                  ${msg.sender === "user"
                    ? "bg-blue-600 text-white ml-auto"
                    : "bg-gray-200 dark:bg-gray-800 text-black dark:text-white mr-auto"}`}>
        {msg.text}
      </div>
    {/each}
  </div>

  <!-- Input Bar -->
  <div class="fixed bottom-0 left-0 w-full md:pl-64 bg-[#0f172a] border-t border-gray-800 p-4 flex items-center gap-3">
    <input
        bind:value={userInput}
        type="text"
        placeholder="Send a message..."
        class="flex-1 bg-[#1e293b] text-white placeholder-gray-400 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        on:keydown={(e) => e.key === "Enter" && sendMessage()}
    />
    <button
        class="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded-lg transition-all"
        on:click={sendMessage}
    >
        Send
    </button>

</div>

</div>
{/if}
