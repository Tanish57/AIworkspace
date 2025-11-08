<script lang="ts">
    import { onMount, tick } from "svelte";
    import { activeSessionId } from "$lib/stores/session";
    import { sendToLlama } from "$lib/services/chat";

    let messages: { sender: "user" | "ai"; text: string; ts?: string }[] = [];
    let userInput = "";
    let loadingHistory = false;
    let started = false;

    let prevSessionId: string | null = null;

    // ✅ Load history ONLY when session ID changes
    $: if ($activeSessionId && $activeSessionId !== prevSessionId) {
        prevSessionId = $activeSessionId;
        loadSessionMessages($activeSessionId);
    }

    async function loadSessionMessages(sessionId: string) {
        loadingHistory = true;
        messages = [];      // ✅ clear old session messages
        started = true;

        const res = await fetch(`http://127.0.0.1:8000/sessions/${sessionId}/messages`);
        const data = await res.json();

        messages = data.map((m: any) => ({
            sender: m.role === "user" ? "user" : "ai",
            text: m.content,
            ts: m.ts
        }));

        loadingHistory = false;

        await tick();
        scrollToBottom();
    }

    async function scrollToBottom() {
        await tick()
        const el = document.getElementById("chat-scroll");
        if (el) {
            el.scrollTop = el.scrollHeight;
            await tick();
            el.scrollTop = el.scrollHeight; // second pass for long content
        }
    }

    async function sendMessage() {
        if (!userInput.trim()) return;

        const sid = $activeSessionId;

        messages = [...messages, { sender: "user", text: userInput }];
        let inputCopy = userInput;
        userInput = "";

        const reply = await sendToLlama(inputCopy, sid);

        messages = [...messages, { sender: "ai", text: reply }];

        await tick();
        scrollToBottom();
    }
</script>

{#if loadingHistory}
<div class="flex h-full items-center justify-center text-gray-400">
    Loading session…
</div>
{:else if !started}
<div class="flex flex-col items-center justify-center h-full text-center space-y-4">
    <h1 class="text-5xl font-bold">TanishGPT</h1>
    <p class="text-gray-600 dark:text-gray-300 max-w-md">
        Your private personal AI assistant with memory — running locally.
    </p>

    <button
        class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        on:click={() => {
            started = true;
            messages = [];
            activeSessionId.set(null); 
        }}
    >
        Start chatting
    </button>
</div>
{:else}
<div class="flex flex-col h-full">

    <div id="chat-scroll" class="flex-1 space-y-4 overflow-y-auto p-2 pb-28">
        {#each messages as msg}
        <div class="w-full">
            <div
                class={`p-3 rounded-lg max-w-xl ${
                    msg.sender === "user"
                        ? "bg-blue-600 text-white ml-auto"
                        : "bg-gray-200 dark:bg-gray-800 text-black dark:text-white mr-auto"
                }`}
            >
                {msg.text}
            </div>
        </div>
        {/each}
    </div>

    <div class="sticky bottom-0 left-0 w-full md:pl-64 bg-[#0f172a] border-t border-gray-800 p-4 flex items-center gap-3">
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
